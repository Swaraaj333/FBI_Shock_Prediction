"""Quick model validation script to confirm training integrity."""
import joblib
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score, classification_report

# Load model and features
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE, "model", "shock_rf_model.pkl"))
features = joblib.load(os.path.join(BASE, "model", "feature_names.pkl"))

print("=" * 60)
print("  MODEL VALIDATION REPORT")
print("=" * 60)

# Model info
print(f"\nModel Type: {type(model).__name__}")
print(f"Features ({len(features)}): {features}")
print(f"Classes: {model.classes_}")
print(f"Max Iterations: {model.max_iter}")
print(f"Max Depth: {model.max_depth}")

# Load processed data and test
df = pd.read_csv(os.path.join(BASE, "data", "processed_mimic.csv"))
stay_count = df["stay_id"].nunique()
print(f"\nTraining Dataset: {len(df):,} rows, {stay_count} ICU stays")

X = df[features].copy()
y = df["Label_Shock_Next"].copy()

# Full dataset accuracy
preds = model.predict(X)
acc = accuracy_score(y, preds)
print(f"\nFull Dataset Accuracy: {acc*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y, preds, target_names=["Stable", "Shock"]))

# Label distribution
stable_n = (y == 0).sum()
shock_n = (y == 1).sum()
print(f"Label Distribution:")
print(f"  Stable (0): {stable_n:,} ({stable_n/len(y)*100:.1f}%)")
print(f"  Shock  (1): {shock_n:,} ({shock_n/len(y)*100:.1f}%)")

# Quick sanity checks
print("\n--- SANITY CHECKS ---")

# Test 1: Healthy vitals should predict STABLE
healthy = pd.DataFrame([{f: np.nan for f in features}])
healthy["HeartRate"] = 72; healthy["SysBP"] = 120; healthy["MAP"] = 85
healthy["RespRate"] = 16; healthy["SpO2"] = 99; healthy["ShockIndex"] = 0.6
healthy["Age"] = 45; healthy["Gender_M"] = 1
p = model.predict(healthy[features])[0]
status = "PASS" if p == 0 else "FAIL"
print(f"  Healthy vitals (HR=72, SysBP=120, SpO2=99) -> {'STABLE' if p==0 else 'SHOCK'} [{status}]")

# Test 2: Shock vitals should predict SHOCK
shock = pd.DataFrame([{f: np.nan for f in features}])
shock["HeartRate"] = 140; shock["SysBP"] = 60; shock["MAP"] = 45
shock["RespRate"] = 30; shock["SpO2"] = 85; shock["ShockIndex"] = 2.33
shock["Age"] = 65; shock["Gender_M"] = 0
p2 = model.predict(shock[features])[0]
status2 = "PASS" if p2 == 1 else "FAIL"
print(f"  Shock vitals (HR=140, SysBP=60, SpO2=85)   -> {'STABLE' if p2==0 else 'SHOCK'} [{status2}]")

# Test 3: Hypoxia alone should trigger
hypox = pd.DataFrame([{f: np.nan for f in features}])
hypox["HeartRate"] = 80; hypox["SysBP"] = 120; hypox["MAP"] = 80
hypox["RespRate"] = 16; hypox["SpO2"] = 77; hypox["ShockIndex"] = 0.67
hypox["Age"] = 50; hypox["Gender_M"] = 1
p3 = model.predict(hypox[features])[0]
status3 = "PASS" if p3 == 1 else "FAIL"
print(f"  Hypoxia only (SpO2=77, everything else ok)  -> {'STABLE' if p3==0 else 'SHOCK'} [{status3}]")

print("\n" + "=" * 60)
print("  MODEL VALIDATION COMPLETE")
print("=" * 60)
