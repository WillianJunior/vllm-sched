#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

# ---------------------------------------------------------------------
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("VLLM_TRAINING_DATA", os.path.join(SCRIPT_DIR, "training_data.txt"))
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------

# The file is tab-separated and uses comma as decimal separator.
#df = pd.read_csv(DATA_FILE, sep="\t", decimal=",")
df = pd.read_csv(DATA_FILE, sep=r"\s+", decimal=".", engine="python")

# Optional: normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

#df = df.iloc[:4000].copy()

if "step" in df.columns:
    df = df.sort_values("step").reset_index(drop=True)

df["batch_size"] = df["decode_reqs"] + df["prefill_reqs"]

# Deriva blocos livres e headroom_ratio (blocos livres por decode ativo)
valid_usage = df["kv_cache_usage"] > 0.05
if valid_usage.any():
    estimated_total_blocks = float((df.loc[valid_usage, "kv_blocks_used"] / df.loc[valid_usage, "kv_cache_usage"]).median())
else:
    estimated_total_blocks = 12787.0

df["free_blocks"] = (estimated_total_blocks - df["kv_blocks_used"]).clip(lower=0)
df["headroom_ratio"] = df["free_blocks"] / df["decode_reqs"].clip(lower=1)

# Backward Labeling: "Última Gota D'água" (Last-Admit Attribution)
# Quando ocorre preempção no passo t, identifica o passo anterior (t-k) de admissão de prefill
# que saturou a memória e atribui a penalidade a ele.
LOOKBACK_WINDOW = int(os.environ.get("VLLM_LOOKBACK_WINDOW", "15"))
preemption_penalty = float(os.environ.get("VLLM_PREEMPTION_PENALTY", "50.0"))

penalized_indices = set()
has_preemption = "num_preemptions" in df.columns

if has_preemption:
    preempt_indices = df.index[df["num_preemptions"] > 0].tolist()
    print(f"\n[Backward-Labeling] Passos com preempção detectados: {len(preempt_indices)}")

    for idx in preempt_indices:
        found = False
        # 1. Procura o prefill mais recente admitido sob risco (headroom_ratio < 1.5 ou kv_cache_usage >= 0.90)
        for k in range(1, LOOKBACK_WINDOW + 1):
            prev_idx = idx - k
            if prev_idx < 0:
                break
            if df.at[prev_idx, "prefill_reqs"] > 0 and (df.at[prev_idx, "headroom_ratio"] < 1.5 or df.at[prev_idx, "kv_cache_usage"] >= 0.90):
                penalized_indices.add(prev_idx)
                found = True
                break

        # 2. Se não achou na zona de 90%, pega a admissão de prefill mais recente na janela
        if not found:
            for k in range(1, LOOKBACK_WINDOW + 1):
                prev_idx = idx - k
                if prev_idx < 0:
                    break
                if df.at[prev_idx, "prefill_reqs"] > 0:
                    penalized_indices.add(prev_idx)
                    break

    print(f"[Backward-Labeling] Passos de admissão penalizados retroativamente ('Última Gota D\'água'): {len(penalized_indices)}")
    df["is_penalized_admission"] = False
    if penalized_indices:
        df.loc[list(penalized_indices), "is_penalized_admission"] = True

# Define o alvo lat_gpu com a penalidade apenas nos passos de admissão culpados
df["target_lat_gpu"] = df["lat_gpu"].copy()
if has_preemption and df["is_penalized_admission"].any():
    penalized_mask = df["is_penalized_admission"]
    df.loc[penalized_mask, "target_lat_gpu"] += preemption_penalty
    print(f"[!] Penalidade retroativa (+{preemption_penalty:.1f} ms) aplicada em {penalized_mask.sum()} passos de admissão de risco.")

df["overhead_ms"] = df["lat_escalonador"] - df["lat_gpu"]

overhead_p95 = df["overhead_ms"].quantile(0.99)
print(f"\n[!] Overhead P95 do sistema: {overhead_p95:.4f} ms")
print(f"[!] Sugestão de corte pro Escalonador: 55ms - {overhead_p95:.4f} ms\n")
# ----------------------------------------------

N_BINS = 20
MAX_PER_BUCKET = 2000
PREEMPTION_QUOTA = int(os.environ.get("VLLM_PREEMPTION_QUOTA", "3000"))

if has_preemption and df["is_penalized_admission"].any():
    df_penalized = df[df["is_penalized_admission"]].copy()
    df_normal = df[~df["is_penalized_admission"]].copy()
    print(f"\n[Estratificação] Passos normais: {len(df_normal)}")
    print(f"[Estratificação] Passos de admissão penalizados: {len(df_penalized)}")

    # 1. Downsampling em buckets apenas para os passos normais
    counts, edges = np.histogram(df_normal["target_lat_gpu"], bins=N_BINS)
    bucket = np.digitize(df_normal["target_lat_gpu"], edges[1:-1], right=False)
    df_normal["bucket"] = bucket

    dfs = []
    for b in range(N_BINS):
        rows = df_normal[df_normal["bucket"] == b]
        if b < 2 and len(rows) > MAX_PER_BUCKET:
            rows = rows.sample(MAX_PER_BUCKET, random_state=42)
        dfs.append(rows)

    df_normal_sampled = pd.concat(dfs).drop(columns="bucket")

    # 2. Amostragem da cota controlada de admissões penalizadas
    if len(df_penalized) > PREEMPTION_QUOTA:
        df_penalized_sampled = df_penalized.sample(PREEMPTION_QUOTA, random_state=42)
    else:
        df_penalized_sampled = df_penalized
    print(f"[Estratificação] Amostras de admissão penalizada selecionadas para treino: {len(df_penalized_sampled)} (de {len(df_penalized)})")
    df = pd.concat([df_normal_sampled, df_penalized_sampled])
else:
    df_normal = df.copy()
    counts, edges = np.histogram(df_normal["target_lat_gpu"], bins=N_BINS)
    bucket = np.digitize(df_normal["target_lat_gpu"], edges[1:-1], right=False)
    df_normal["bucket"] = bucket
    dfs = []
    for b in range(N_BINS):
        rows = df_normal[df_normal["bucket"] == b]
        if b < 2 and len(rows) > MAX_PER_BUCKET:
            rows = rows.sample(MAX_PER_BUCKET, random_state=42)
        dfs.append(rows)
    df = pd.concat(dfs).drop(columns="bucket")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFiltered dataset size: {len(df)}")
if has_preemption and "is_penalized_admission" in df.columns:
    preempt_pct = df["is_penalized_admission"].mean() * 100
    print(f"[Estratificação] Proporção de admissões penalizadas no dataset final: {preempt_pct:.2f}% ({df['is_penalized_admission'].sum()} amostras)")

print(df)

# Features (7 features incluindo headroom_ratio) e target
X = df[["token_budget", "decode_reqs", "prefill_reqs", "kv_blocks_used", "batch_size", "kv_cache_usage", "headroom_ratio"]]
y = df["target_lat_gpu"].copy()

N_BINS = 20

bins = pd.cut(df["lat_gpu"], bins=N_BINS)

hist = bins.value_counts(sort=False)

for interval, count in hist.items():
    print(f"{interval}: {count}")

# ---------------------------------------------------------------------
# Split data
# ---------------------------------------------------------------------
# 10% test
# 20% validation
# 70% training
# ---------------------------------------------------------------------

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=RANDOM_STATE,
)

# Validation should be 20% of the total dataset.
# Since trainval contains 90%, split off 20/90 = 2/9.
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=2/9,
    random_state=RANDOM_STATE,
)

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples      : {len(X_test)}")

# ---------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------

regressor = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=16,
    max_depth=16,
    min_child_samples=50,
    random_state=RANDOM_STATE,
)

regressor.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
)

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

y_pred = regressor.predict(X_test)
y_pred_val = regressor.predict(X_val)

mae = mean_absolute_error(y_test, y_pred)
mae_val = mean_absolute_error(y_val, y_pred_val)

print(f"\nVal  MAE: {mae_val:.8f}")
print(f"Test MAE: {mae:.8f}")

# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------

print("\nFeature importance:")
for name, importance in zip(X.columns, regressor.feature_importances_):
    print(f"{name:15s}: {importance}")

print(f"Mean latency: {y_test.mean():.4f}")
print(f"MAE: {mae:.4f}")
print(f"Relative MAE: {mae / y_test.mean():.4f}")

# Exporta o Regressor
import json
target_gpu_cutoff = 33.0

joblib.dump(regressor, os.path.join(SCRIPT_DIR, 'lgbm_oraculo.pkl'))
print(f"\n[!] Regressor exportado com sucesso: {os.path.join(SCRIPT_DIR, 'lgbm_oraculo.pkl')}")

meta = {
    "model_type": "LGBMRegressor",
    "overhead_p95": float(overhead_p95),
    "suggested_gpu_cutoff": target_gpu_cutoff,
    "test_mae": float(mae),
    "mean_latency": float(y_test.mean()),
    "features": list(X.columns)
}

meta_path = os.path.join(SCRIPT_DIR, "lgbm_oraculo_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)
print(f"[!] Metadados exportados com sucesso: {meta_path} (Corte sugerido na GPU: {target_gpu_cutoff:.4f} ms)")

with open(os.path.join(SCRIPT_DIR, "lgbm_model.pkl"), "wb") as f:
    pickle.dump(regressor, f)

# === classifier (opcional para análise) =====================================

from lightgbm import LGBMClassifier

classifier = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=32,
    random_state=42,
    class_weight="balanced",
)

limite_gpu_ms = target_gpu_cutoff

print(f"\n[!] Treinando Classificador com limite de corte na GPU de: {limite_gpu_ms:.4f} ms")

df["is_over_slo"] = (y > limite_gpu_ms)
y_cls = df["is_over_slo"]

X_trainval, X_test_cls, y_trainval_cls, y_test_cls = train_test_split(
    X,
    y_cls,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y_cls
)

X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(
    X_trainval,
    y_trainval_cls,
    test_size=2/9,
    random_state=RANDOM_STATE,
    stratify=y_trainval_cls
)

classifier.fit(X_train_cls, y_train_cls)

y_pred_cls = classifier.predict(X_test_cls)

from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test_cls, y_pred_cls))
print(classification_report(y_test_cls, y_pred_cls))

# Salva classificador separadamente caso desejado para inspeção
joblib.dump(classifier, os.path.join(SCRIPT_DIR, 'lgbm_classificador.pkl'))