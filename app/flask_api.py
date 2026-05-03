"""
flask_api.py
============
REST API backend for Hemodynamic Shock Prediction.

This service loads a pre-trained HistGradientBoostingClassifier and exposes
endpoints for real-time and batch shock risk prediction. The model was trained
on MIMIC-IV ICU data and predicts whether a patient will enter hemodynamic
shock within the next 1-2 hours based on current vitals.

Endpoints:
    POST /predict       - Single-row JSON prediction
    POST /predict_csv   - Batch CSV upload with temporal feature engineering
    GET  /health        - Liveness probe
    GET  /model_info    - Model metadata (features, classes, hyperparams)

Usage:
    python app/flask_api.py
    -> http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import traceback

app = Flask(__name__)
CORS(app)  # Streamlit runs on a different port, so CORS is required

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────

# Resolve paths relative to project root, not CWD, so the API works
# regardless of where it's invoked from (e.g., `python app/flask_api.py`)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_artifacts():
    """
    Load the serialized model and feature list from disk.

    Returns:
        tuple: (model, None, feature_names)
               The None is a legacy slot for an imputer we no longer use
               since HistGradientBoosting handles NaNs natively.
    """
    model_path = os.path.join(MODEL_DIR, "shock_rf_model.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if not all(os.path.exists(p) for p in [model_path, features_path]):
        raise FileNotFoundError(
            "Model files not found! Run 'python ml_pipeline/train_model.py' first."
        )

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    return model, None, feature_names


# Load once at startup — fail loudly if the model isn't there
try:
    MODEL, IMPUTER, FEATURE_NAMES = load_artifacts()
    print(f"[OK] Model loaded. Features: {FEATURE_NAMES}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    MODEL, IMPUTER, FEATURE_NAMES = None, None, None


# ──────────────────────────────────────────────
# Prediction Helpers
# ──────────────────────────────────────────────

def prepare_input(data_dict):
    """
    Normalize and enrich a single row of vitals for model input.

    Handles three things:
      1. Maps common field name variants (e.g., 'heart_rate' -> 'HeartRate')
         so the API is flexible about what callers send us.
      2. Computes ShockIndex (HR/SysBP) if not already provided.
      3. Fills any missing model features with NaN — the model handles
         NaN natively, so no imputation is needed.
    """
    row = {}

    # Flexible field name mapping — allows the API to accept multiple
    # naming conventions without forcing callers to match exactly.
    field_map = {
        "HeartRate": "HeartRate",
        "heart_rate": "HeartRate",
        "hr": "HeartRate",
        "SysBP": "SysBP",
        "sys_bp": "SysBP",
        "systolic_bp": "SysBP",
        "MAP": "MAP",
        "map": "MAP",
        "mean_arterial_pressure": "MAP",
        "RespRate": "RespRate",
        "resp_rate": "RespRate",
        "respiratory_rate": "RespRate",
        "SpO2": "SpO2",
        "spo2": "SpO2",
        "oxygen_saturation": "SpO2",
        "Age": "Age",
        "age": "Age",
        "Gender_M": "Gender_M",
        "gender_m": "Gender_M",
        "ShockIndex": "ShockIndex",
        "shock_index": "ShockIndex",
        "HR_Change": "HR_Change",
        "SysBP_Change": "SysBP_Change",
        "MAP_Change": "MAP_Change",
        "SpO2_Change": "SpO2_Change",
    }

    for key, val in data_dict.items():
        mapped = field_map.get(key, key)
        if mapped in FEATURE_NAMES:
            try:
                row[mapped] = float(val) if val is not None and val != "" else np.nan
            except (ValueError, TypeError):
                row[mapped] = np.nan

    # Derive ShockIndex when not explicitly provided — this is a key
    # clinical indicator (HR/SysBP > 1.0 strongly suggests shock).
    if "ShockIndex" not in row or np.isnan(row.get("ShockIndex", np.nan)):
        hr = row.get("HeartRate", np.nan)
        sbp = row.get("SysBP", np.nan)
        if not np.isnan(hr) and not np.isnan(sbp) and sbp > 0:
            row["ShockIndex"] = round(hr / sbp, 3)

    # Pad any features the caller didn't send with NaN
    for feat in FEATURE_NAMES:
        if feat not in row:
            row[feat] = np.nan

    return row


def predict_single(row_dict):
    """
    Run inference on a single set of vitals.

    Returns a dict with the binary prediction, human-readable label,
    class probabilities, and the computed shock index.
    """
    prepared = prepare_input(row_dict)

    # Column order must match what the model was trained on
    df = pd.DataFrame([prepared])[FEATURE_NAMES]

    # HistGradientBoosting handles NaNs natively — no imputation step
    df_imputed = df

    prediction = int(MODEL.predict(df_imputed)[0])
    probabilities = MODEL.predict_proba(df_imputed)[0]

    return {
        "prediction": prediction,
        "label": "SHOCK" if prediction == 1 else "STABLE",
        "confidence": round(float(max(probabilities)) * 100, 1),
        "prob_stable": round(float(probabilities[0]) * 100, 1),
        "prob_shock": round(float(probabilities[1]) * 100, 1),
        "shock_index": prepared.get("ShockIndex"),
        "input_values": {k: v for k, v in prepared.items() if not np.isnan(v) if isinstance(v, (int, float))},
    }


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health_check():
    """Liveness check — returns model load status."""
    return jsonify({
        "status": "healthy" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL is not None,
    })


@app.route("/model_info", methods=["GET"])
def model_info():
    """Returns model metadata so the frontend can display model details."""
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify({
        "model_type": "HistGradientBoostingClassifier",
        "n_estimators": getattr(MODEL, 'max_iter', 100),
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "classes": ["Stable (0)", "Shock (1)"],
        "description": "Hemodynamic Shock Predictor trained on real MIMIC-IV ICU data",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single-row prediction from a JSON payload.

    Expects vitals like HeartRate, SysBP, MAP, RespRate, SpO2, etc.
    Returns the prediction label, confidence, and probabilities.
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        result = predict_single(data)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    """
    Batch prediction from a CSV file upload.

    This endpoint replicates the training pipeline's preprocessing:
      1. Forward-fills missing vitals (limit=2 to match training)
      2. Computes ShockIndex from HR and SysBP
      3. Computes temporal change features (diff between consecutive rows)
      4. Runs each row through the model

    This temporal awareness is critical — without it, the model would
    treat each row independently and miss deterioration trends.
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use 'file' field."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        df = pd.read_csv(file)
        print(f"[OK] Received CSV with {len(df)} rows, columns: {list(df.columns)}")

        # Step 1: Forward-fill gaps in vitals (simulates sensor reconnection).
        # limit=2 matches the training pipeline to avoid stale data leaking
        # into predictions many hours after a sensor disconnected.
        vitals_cols = [c for c in df.columns if c not in ['stay_id', 'subject_id', 'charttime']]
        if "stay_id" in df.columns:
            df[vitals_cols] = df.groupby("stay_id")[vitals_cols].ffill(limit=2)
        else:
            df[vitals_cols] = df[vitals_cols].ffill(limit=2)

        # Step 2: Compute ShockIndex at the DataFrame level for efficiency
        if 'HeartRate' in df.columns and 'SysBP' in df.columns:
            df['ShockIndex'] = np.where(
                (df['SysBP'].notna()) & (df['SysBP'] > 0),
                np.round(df['HeartRate'] / df['SysBP'], 3),
                np.nan
            )

        # Step 3: Temporal change features — the model relies on these
        # to detect rapid deterioration (e.g., HR spiking, BP dropping)
        change_map = {'HeartRate': 'HR_Change', 'SysBP': 'SysBP_Change', 'MAP': 'MAP_Change', 'SpO2': 'SpO2_Change'}
        for col, change_col in change_map.items():
            if col in df.columns:
                if 'stay_id' in df.columns:
                    df[change_col] = df.groupby('stay_id')[col].diff()
                else:
                    df[change_col] = df[col].diff()

        # Step 4: Predict each row individually
        results = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            pred = predict_single(row_dict)
            pred["row_index"] = idx
            results.append(pred)

        # Aggregate stats for the frontend banner logic
        n_shock = sum(1 for r in results if r["prediction"] == 1)
        n_stable = len(results) - n_shock

        # Any shock prediction in the timeline = CRITICAL_RISK
        patient_status = "CRITICAL_RISK" if n_shock > 0 else "STABLE"

        return jsonify({
            "total_rows": len(results),
            "shock_count": n_shock,
            "stable_count": n_stable,
            "shock_percentage": round(n_shock / len(results) * 100, 1) if results else 0,
            "patient_status": patient_status,
            "predictions": results,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Shock Prediction API")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
