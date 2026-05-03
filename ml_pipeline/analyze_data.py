import pandas as pd
import numpy as np
import joblib

model = joblib.load("model/shock_rf_model.pkl")
features = joblib.load("model/feature_names.pkl")

# Load demo CSV 
df = pd.read_csv("../../mimic_patient_demo.csv")

# Forward-fill (as the API does)
df = df.groupby("stay_id", group_keys=False).apply(lambda g: g.ffill())

# Find clinically concerning rows
print("=== CLINICALLY CONCERNING ROWS ===")
print("\nRows where SpO2 < 90:")
mask_spo2 = df['SpO2'] < 90
print(df[mask_spo2][['charttime','HeartRate','MAP','SysBP','RespRate','SpO2']])

print("\nRows where MAP < 65:")
if 'MAP' in df.columns:
    mask_map = df['MAP'] < 65
    print(df[mask_map][['charttime','HeartRate','MAP','SysBP','RespRate','SpO2']])

print("\nRows where RespRate > 25 & HeartRate > 110:")
mask_resp = (df['RespRate'] > 25) & (df['HeartRate'] > 110)
print(df[mask_resp][['charttime','HeartRate','MAP','SysBP','RespRate','SpO2']])

# Now build predictions exactly as the API does
rows = []
for idx, row in df.iterrows():
    r = {}
    for f in features:
        val = row.get(f, np.nan)
        r[f] = float(val) if pd.notna(val) else np.nan
    
    if pd.isna(r.get("ShockIndex")):
        hr = r.get("HeartRate")
        sbp = r.get("SysBP")
        if pd.notna(hr) and pd.notna(sbp) and sbp > 0:
            r["ShockIndex"] = round(hr / sbp, 3)
    
    rows.append(r)

X_test = pd.DataFrame(rows)[features]
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

df["Pred"] = preds
df["Prob_Shock"] = probs[:, 1]

print("\n=== PREDICTION RESULTS FOR CONCERNING ROWS ===")
print("\nSpO2 < 90 rows prediction:")
print(df[mask_spo2][['charttime','HeartRate','MAP','SysBP','RespRate','SpO2','Pred','Prob_Shock']])

print(f"\nTotal SHOCK predictions: {(preds==1).sum()} / {len(preds)}")
print(f"Total STABLE predictions: {(preds==0).sum()} / {len(preds)}")

# Check what the model features look like for SpO2=77 row
spo2_77_idx = df[df['SpO2'] == 77].index
if len(spo2_77_idx) > 0:
    print("\n=== MODEL INPUT FOR SpO2=77 ROW ===")
    print(X_test.loc[spo2_77_idx[0]])
    print(f"Prediction: {preds[spo2_77_idx[0]]}, Prob_Shock: {probs[spo2_77_idx[0]][1]:.4f}")

# Check Age/Gender NaN impact - the training data NEVER has NaN for Age/Gender
print("\n=== NAN ANALYSIS ===")
print("Features with NaN in demo CSV predictions:")
nan_counts = X_test.isna().sum()
print(nan_counts[nan_counts > 0])

# Check the training data
train_df = pd.read_csv("data/processed_mimic.csv")
print("\nTraining data NaN counts per feature:")
train_nan = train_df[features].isna().sum()
print(train_nan)
