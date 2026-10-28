#!/usr/bin/env python3

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import os
import json

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.environ.get("VLLM_TRAINING_DATA", os.path.join(SCRIPT_DIR, "training_data.txt"))
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------

df = pd.read_csv(DATA_FILE, sep=r"\s+", decimal=".", engine="python")

# Optional: normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

df["batch_size"] = df["decode_reqs"] + df["prefill_reqs"]

df["overhead_ms"] = df["lat_escalonador"] - df["lat_gpu"]

overhead_p95 = df["overhead_ms"].quantile(0.99)
print(f"\n[!] Overhead P95 do sistema: {overhead_p95:.4f} ms")
print(f"[!] Sugestão de corte pro Escalonador: 55ms - {overhead_p95:.4f} ms\n")
# ----------------------------------------------

print(df)

N_BINS = 20
MAX_PER_BUCKET = 2000

# Compute bucket edges
counts, edges = np.histogram(df["lat_gpu"], bins=N_BINS)

# Assign each row to a bucket (0..19)
bucket = np.digitize(df["lat_gpu"], edges[1:-1], right=False)
df["bucket"] = bucket

# Keep all rows except cap buckets 0 and 1
dfs = []

for b in range(N_BINS):
    rows = df[df["bucket"] == b]

    if b < 2 and len(rows) > MAX_PER_BUCKET:
        rows = rows.sample(MAX_PER_BUCKET, random_state=42)

    dfs.append(rows)

df = (
    pd.concat(dfs)
      .drop(columns="bucket")
      .sample(frac=1, random_state=42)   # shuffle
      .reset_index(drop=True)
)

print(f"Filtered dataset size: {len(df)}")
print(df)

# Features and target (5 features do regressor clássico)
X = df[["token_budget", "decode_reqs", "prefill_reqs", "kv_blocks_used", "batch_size"]]
y = df["lat_gpu"].copy()

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
# Train Regressor Model
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

# Exporta o Regressor (tanto com sufixo _antigo quanto o oraculo padrão)
target_gpu_cutoff = 33.0

out_antigo = os.path.join(SCRIPT_DIR, 'lgbm_oraculo_antigo.pkl')
joblib.dump(regressor, out_antigo)
print(f"\n[!] Regressor antigo exportado com sucesso: {out_antigo}")

meta = {
    "model_type": "LGBMRegressor",
    "overhead_p95": float(overhead_p95),
    "suggested_gpu_cutoff": target_gpu_cutoff,
    "test_mae": float(mae),
    "mean_latency": float(y_test.mean()),
    "features": list(X.columns)
}

meta_path = os.path.join(SCRIPT_DIR, "lgbm_oraculo_antigo_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)
print(f"[!] Metadados exportados com sucesso: {meta_path} (Corte sugerido na GPU: {target_gpu_cutoff:.4f} ms)")

with open(os.path.join(SCRIPT_DIR, "lgbm_model_antigo.pkl"), "wb") as f:
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

df["is_over_slo"] = (df["lat_gpu"] > limite_gpu_ms)
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

joblib.dump(classifier, os.path.join(SCRIPT_DIR, 'lgbm_classificador_antigo.pkl'))
