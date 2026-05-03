"""
test_app.py
===========
End-to-end validation suite for the Flask prediction API.

Tests cover three layers:
  1. System health — model loads correctly, endpoints respond
  2. Single prediction — known-good stable and shock payloads
  3. Batch CSV integration — all 3 test CSVs produce expected clinical outcomes

Run with: pytest tests/test_app.py -v
"""

import sys
import os
import json
import pytest

# Add project root to path so we can import from the app/ package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.flask_api import app, MODEL

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_data')

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Verify the API is live and the model loaded successfully at startup."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert MODEL is not None

def test_model_info(client):
    """Confirm /model_info returns the correct model type and feature list."""
    response = client.get('/model_info')
    assert response.status_code == 200
    data = response.get_json()
    assert data['model_type'] == "HistGradientBoostingClassifier"
    assert "features" in data
    assert len(data['features']) > 0

def test_predict_single_stable(client):
    """A textbook-healthy patient should always predict STABLE (no false alarms)."""
    vitals = {
        "HeartRate": 70,
        "SysBP": 120,
        "MAP": 85,
        "RespRate": 16,
        "SpO2": 99,
        "Age": 45,
        "Gender_M": 1
    }
    response = client.post('/predict', json=vitals)
    assert response.status_code == 200
    data = response.get_json()
    assert data['prediction'] == 0
    assert data['label'] == 'STABLE'

def test_predict_single_shock(client):
    """Classic shock vitals (high HR, low BP, low SpO2) must trigger SHOCK."""
    vitals = {
        "HeartRate": 140,
        "SysBP": 60,
        "MAP": 45,
        "RespRate": 30,
        "SpO2": 85,
        "Age": 65,
        "Gender_M": 0
    }
    response = client.post('/predict', json=vitals)
    assert response.status_code == 200
    data = response.get_json()
    assert data['prediction'] == 1
    assert data['label'] == 'SHOCK'

def test_predict_csv_stable(client):
    """A stable patient timeline should produce zero shock flags."""
    file_path = os.path.join(TEST_DATA_DIR, "patient_stable.csv")
    assert os.path.exists(file_path), f"{file_path} not found"
    
    with open(file_path, "rb") as f:
        data = {
            "file": (f, "patient_stable.csv")
        }
        response = client.post('/predict_csv', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    result = response.get_json()
    assert result['patient_status'] == 'STABLE'
    assert result['shock_count'] == 0

def test_predict_csv_crash(client):
    """A deteriorating patient must be flagged CRITICAL_RISK with the crash rows identified."""
    file_path = os.path.join(TEST_DATA_DIR, "patient_sudden_crash.csv")
    assert os.path.exists(file_path), f"{file_path} not found"
    
    with open(file_path, "rb") as f:
        data = {
            "file": (f, "patient_sudden_crash.csv")
        }
        response = client.post('/predict_csv', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    result = response.get_json()
    assert result['patient_status'] == 'CRITICAL_RISK'
    assert result['shock_count'] > 0
    
    # The last row has the worst vitals — it must be flagged
    predictions = result['predictions']
    last_pred = predictions[-1]
    assert last_pred['prediction'] == 1

def test_predict_csv_missing_data(client):
    """Blank rows in the CSV should be forward-filled, not cause a 500 error."""
    file_path = os.path.join(TEST_DATA_DIR, "patient_missing_data.csv")
    assert os.path.exists(file_path), f"{file_path} not found"
    
    with open(file_path, "rb") as f:
        data = {
            "file": (f, "patient_missing_data.csv")
        }
        response = client.post('/predict_csv', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    result = response.get_json()
    
    predictions = result['predictions']
    assert len(predictions) > 0
    
    # Ensure missing data didn't cause 500 error and we got valid predictions
    assert "predictions" in result
    
    # Row 3 (index 2 in 0-based CSV, index 3 in predictions) is completely
    # blank in the CSV. After ffill, it should have inherited HeartRate
    # from the previous row.
    row_4 = predictions[3]
    assert "HeartRate" in row_4["input_values"], "ffill failed — blank row has no HeartRate"
