#!/usr/bin/env python3

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_FILE = "lats_test_data.txt"   # Change to your filename
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------

# The file is tab-separated and uses comma as decimal separator.
#df = pd.read_csv(DATA_FILE, sep="\t", decimal=",")
df = pd.read_csv(DATA_FILE, sep=r"\s+", decimal=",", engine="python")

# Optional: normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

#df = df.iloc[:4000].copy()

df["batch_size"] = df["decode_reqs"] + df["prefill_reqs"]

# Data is shifted...
df["token_budget"] = df["token_budget"].shift(1)

N_BINS = 20
MAX_PER_BUCKET = 2000

# Compute bucket edges
counts, edges = np.histogram(df["lat"], bins=N_BINS)

# Assign each row to a bucket (0..19)
bucket = np.digitize(df["lat"], edges[1:-1], right=False)
df["bucket"] = bucket

# Keep all rows except cap buckets 0 and 1
dfs = []

for b in range(N_BINS):
    rows = df[df["bucket"] == b]

    if b < 2 and len(rows) > MAX_PER_BUCKET:
        rows = rows.sample(MAX_PER_BUCKET, random_state=42)

    dfs.append(rows)

df_orig = df.copy()
df = (
    pd.concat(dfs)
      .drop(columns="bucket")
      .sample(frac=1, random_state=42)   # shuffle
      .reset_index(drop=True)
)

print(f"Filtered dataset size: {len(df)}")


print(df)

# Features and target
X = df[["token_budget", "decode_reqs", "prefill_reqs", "kv_blocks_used", "batch_size"]]
#X = df[["token_budget"]]


y = df["lat"]
#y = df["is_over_slo"]

N_BINS = 20

bins = pd.cut(df["lat"], bins=N_BINS)

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

model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=16,
    max_depth=16,
    min_child_samples=50,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
)

# ---------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------

y_pred = model.predict(X_test)
y_pred_val = model.predict(X_val)

mae = mean_absolute_error(y_test, y_pred)
mae_val = mean_absolute_error(y_val, y_pred_val)

print(f"\nVal  MAE: {mae_val:.8f}")
print(f"Test MAE: {mae:.8f}")

# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------

print("\nFeature importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name:15s}: {importance}")

print(f"Mean latency: {y_test.mean():.4f}")
print(f"MAE: {mae:.4f}")
print(f"Relative MAE: {mae / y_test.mean():.4f}")







# === classifier =====================================

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=32,
    random_state=42,
    #class_weight={False: 1, True: 2},
    class_weight="balanced",
)

df["is_over_slo"] = df["lat"] > 0.050
y = df["is_over_slo"]

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


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance)


importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.booster_.feature_importance(importance_type="gain")
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\ngain importance ------------------------")
print(importance)
