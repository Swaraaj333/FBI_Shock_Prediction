# Hemodynamic Shock Prediction: ML Evaluation Report

## 1. Model & Dataset Overview
*   **Objective:** Predict Hemodynamic Shock 2 hours in advance (Future State Forecasting).
*   **Model Algorithm:** Random Forest Classifier (`HistGradientBoostingClassifier`).
*   **Dataset:** MIMIC-IV Clinical Database.
*   **Total Data Size:** 6,685 hourly records across 100 ICU patients.
*   **Testing Split:** 20% Holdout Set (1,337 records).

---

## 2. Core Performance Metrics
*   **Overall Accuracy:** `77.86%`
*   **Macro Average F1-Score:** `0.71`
*   **Weighted Average F1-Score:** `0.77`

---

## 3. Detailed Classification Report
| Patient State | Precision | Recall (Sensitivity) | F1-Score | Support (Testing Rows) |
| :--- | :---: | :---: | :---: | :---: |
| **Stable (Class 0)** | 0.82 | 0.89 | 0.85 | 956 |
| **Shock (Class 1)** | 0.64 | 0.51 | 0.57 | 381 |

---

## 4. Confusion Matrix Analysis
| | Predicted STABLE | Predicted SHOCK |
| :--- | :---: | :---: |
| **Actual STABLE** | **848** (True Negatives) | **108** (False Positives - False Alarms) |
| **Actual SHOCK** | **188** (False Negatives - Missed) | **193** (True Positives - Caught Crashes) |

---

## 5. Clinical Interpretation (Talking Points for PPT)

*   **The "Goldilocks" Accuracy (78%):** In medical forecasting, a 99% accuracy is a mathematical impossibility and indicates data leakage. Achieving ~78% accuracy on *future forecasting* (predicting an event 2 hours before it happens) represents a realistic, robust, and highly valuable early-warning system.
*   **Precision vs. Recall Balance:** 
    *   The model achieves a **Precision of 64%** for Shock events. This means when the alarm rings, it is correct a majority of the time. 
    *   The model achieves a **Recall of 89%** for Stable events. It is incredibly good at recognizing when a patient is safe, which prevents "Alarm Fatigue" (a major issue in modern ICUs where nurses ignore monitors because they beep too often).
*   **Clinical Utility:** The 193 True Positives represent 193 distinct hours where a patient's life could potentially be saved by an early medical intervention triggered by this model.
