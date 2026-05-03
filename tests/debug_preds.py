import pandas as pd
import numpy as np
import joblib

def test():
    # Load model
    model = joblib.load("model/shock_rf_model.pkl")
    features = joblib.load("model/feature_names.pkl")
    
    # Load test data
    df = pd.read_csv("../../mimic_patient_demo.csv")
    
    vitals_cols = [c for c in df.columns if c not in ['stay_id', 'subject_id', 'charttime']]
    if "stay_id" in df.columns:
        df[vitals_cols] = df.groupby("stay_id")[vitals_cols].ffill(limit=2)
    else:
        df[vitals_cols] = df[vitals_cols].ffill(limit=2)
        
    print(f"Row 16 (SpO2=77) after ffill:")
    row16 = df.iloc[16]
    print(row16)

    if 'HeartRate' in df.columns and 'SysBP' in df.columns:
        df['ShockIndex'] = np.where(
            (df['SysBP'].notna()) & (df['SysBP'] > 0),
            np.round(df['HeartRate'] / df['SysBP'], 3),
            np.nan
        )

    change_map = {'HeartRate': 'HR_Change', 'SysBP': 'SysBP_Change', 'MAP': 'MAP_Change', 'SpO2': 'SpO2_Change'}
    for col, change_col in change_map.items():
        if col in df.columns:
            if 'stay_id' in df.columns:
                df[change_col] = df.groupby('stay_id')[col].diff()
            else:
                df[change_col] = df[col].diff()
    
    # Create prediction df
    rows = []
    for idx, row in df.iterrows():
        r = {}
        for f in features:
            val = row.get(f, np.nan)
            r[f] = float(val) if pd.notna(val) else np.nan
        rows.append(r)
        
    X_test = pd.DataFrame(rows)[features]
    
    print(f"Row 16 parsed for model:")
    print(X_test.iloc[16])
    
    # Let's test a synthetic row!
    syn_row = X_test.iloc[16].copy()
    syn_row["SpO2"] = 77.0
    syn_row["SysBP"] = 120.0
    syn_row["MAP"] = 80.0
    syn_row["HeartRate"] = 80.0
    syn_row["RespRate"] = 16.0
    syn_pred = model.predict(pd.DataFrame([syn_row]))[0]
    syn_prob = model.predict_proba(pd.DataFrame([syn_row]))[0]
    print(f"\nSynthetic row with SpO2=77, and everything else normal. Pred: {syn_pred}, Prob: {syn_prob}")
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    df["Pred"] = preds
    
    print(f"\nActual prediction for Row 16: Pred={preds[16]}, Prob={probs[16]}")
    
    print("\nPredicted as Shock:")
    print(df[df["Pred"] == 1][["HeartRate", "SysBP", "MAP", "SpO2", "RespRate", "Pred"]])

if __name__ == "__main__":
    test()
