"""
train_model.py
==============
Trains a HistGradientBoostingClassifier on the processed MIMIC-IV dataset
for hemodynamic shock prediction (1-2 hour lookahead).

We use HistGradientBoosting instead of RandomForest because it handles
NaN values natively — no imputation needed. This is critical for ICU data
where sensor dropouts create legitimate missing values.

Prerequisites:
  Run `python ml_pipeline/process_mimic_data.py` first to generate
  data/processed_mimic.csv

Usage:
  python ml_pipeline/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def train_and_save_model():
    """
    Loads the processed MIMIC-IV dataset, trains a Random Forest model,
    evaluates it, and saves it for the Flask API.
    """

    print("=" * 60)
    print("  Hemodynamic Shock Prediction - Model Training")
    print("=" * 60)

    # -- Step 1: Load Data --
    print("\n[1/5] Loading the processed MIMIC-IV dataset...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "processed_mimic.csv")
    if not os.path.exists(data_path):
        print(f"  ERROR: '{data_path}' not found!")
        print("  Please run 'python process_mimic_data.py' first.")
        return

    df = pd.read_csv(data_path)
    print(f"  [OK] Loaded {len(df):,} rows x {df.shape[1]} columns")
    print(f"  [OK] Unique ICU stays: {df['stay_id'].nunique()}")

    # -- Step 2: Prepare Features & Labels --
    print("\n[2/5] Preparing features and labels...")

    # Features we'll use for prediction
    feature_columns = [
        "HeartRate", "SysBP", "MAP", "RespRate", "SpO2",
        "ShockIndex",
        "HR_Change", "SysBP_Change", "MAP_Change", "SpO2_Change",
        "Age", "Gender_M",
    ]

    # Only use features that exist in the dataset
    available_features = [c for c in feature_columns if c in df.columns]
    print(f"  [OK] Using {len(available_features)} features: {available_features}")

    X = df[available_features].copy()
    y = df["Label_Shock_Next"].copy()

    # Intentionally mask 15% of demographics to train the model to handle
    # missing Age/Gender gracefully — in production, these may not always
    # be available at prediction time.
    mask_rate = 0.15
    rng = np.random.RandomState(42)

    for col in ['Age', 'Gender_M']:
        if col in X.columns:
            mask = rng.random(len(X)) < mask_rate
            X.loc[mask, col] = np.nan

    X_imputed = X

    # Sample weights compensate for class imbalance in critical signals.
    # Hypoxia (SpO2 < 90) is rare (~56 rows in 6,685) but clinically
    # life-threatening, so we upweight it 50x to force the model to learn it.
    # Severe respiratory distress gets 20x for similar reasons.
    sample_weights = np.ones(len(X_imputed))
    if 'SpO2' in X_imputed.columns:
        sample_weights[X_imputed['SpO2'] < 90] = 50.0
    if 'RespRate' in X_imputed.columns and 'HeartRate' in X_imputed.columns:
        severe_resp = (X_imputed['RespRate'] > 25) & (X_imputed['HeartRate'] > 110)
        sample_weights[severe_resp] = 20.0

    print(f"\n  -- Label Distribution --")
    print(f"    Stable (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
    print(f"    Shock  (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")

    # -- Step 3: Train/Test Split --
    print("\n[3/5] Splitting into train (80%) and test (20%) sets...")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_imputed, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  [OK] Training set: {len(X_train):,} rows")
    print(f"  [OK] Testing set:  {len(X_test):,} rows")

    # -- Step 4: Train Model --
    print("\n[4/5] Training HistGradientBoostingClassifier...")

    # max_depth=12 is deliberately deep — ICU shock patterns involve
    # complex feature interactions (e.g., HR spike + BP drop + SpO2 drop).
    # min_samples_leaf=5 prevents overfitting on rare edge cases.
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
    )

    # Use sample weights instead of class_weight="balanced"
    model.fit(X_train, y_train, sample_weight=w_train)
    print("  [OK] Model trained successfully!")

    # -- Step 5: Evaluate --
    print("\n[5/5] Evaluating model performance...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  ==================================")
    print(f"  Model Accuracy: {accuracy * 100:.2f}%")
    print(f"  ==================================")

    print(f"\n  -- Classification Report --")
    print(classification_report(y_test, y_pred, target_names=["Stable", "Shock"]))

    print(f"  -- Confusion Matrix --")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Stable  Shock")
    print(f"    Actual Stable  {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"    Actual Shock   {cm[1][0]:>5}  {cm[1][1]:>5}")

    # -- Feature Importance --
    print(f"\n  -- Feature Importance --")
    print("  (HistGradientBoostingClassifier does not expose native feature_importances_)")

    # -- Save Model --
    model_dir = os.path.join(BASE_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "shock_rf_model.pkl"))
    joblib.dump(available_features, os.path.join(model_dir, "feature_names.pkl"))

    print(f"\n  [OK] Model saved  -> {os.path.join('model', 'shock_rf_model.pkl')}")
    print(f"  [OK] Feature list  -> {os.path.join('model', 'feature_names.pkl')}")

    print("\n" + "=" * 60)
    print("  DONE! Training complete. Model ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()
