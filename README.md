# 🏥 Hemodynamic Shock Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Flask](https://img.shields.io/badge/Flask-API-black)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)

A machine learning application designed to predict **future Hemodynamic Shock risk** in ICU patients 1-2 hours before it happens. Built using real MIMIC-IV hospital data!

*(Drop a cool screenshot of your Streamlit UI here!)*
<!-- ![UI Screenshot](path/to/your/screenshot.png) -->

---

## 🎯 Project Objective
The goal of this project is to create an early-warning system for hospital ICUs. Instead of just sounding an alarm *after* a patient crashes, this application uses a Random Forest Machine Learning model to analyze a patient's vitals (Heart Rate, Blood Pressure, SpO2, etc.) and predict if they are going to go into Hemodynamic Shock in the near future. 

## 💻 UI Interface & Features
The application has a clean, easy-to-use Streamlit web interface with three main tabs:
1. **CSV Upload:** Upload a timeline of patient vitals. The app will process the data and instantly flash a massive Red Warning Banner if it detects a future crash, telling you exactly which hours the patient is at risk.
2. **Manual Entry:** A quick "pocket calculator" for nurses. Manually punch in a patient's current vitals to get an instant snapshot of their risk level.
3. **Data Explorer:** Beautiful, interactive graphs that show you exactly how the model works. You can view Label Distributions, Vital Sign Histograms, Feature Correlations, and an interactive Patient Timeline to see how vitals change during a crash.

---

## 🚀 How to Operate the Application

To get this project running on your local machine, you need to start both the Flask backend (which runs the ML model) and the Streamlit frontend (the user interface).

**1. Install Dependencies**
Make sure you have all the required libraries installed:
```bash
pip install pandas numpy scikit-learn flask streamlit plotly
```

**2. Start the Backend API**
Open a terminal and run the Flask API:
```bash
python app/flask_api.py
```
*(Leave this terminal running in the background!)*

**3. Start the Frontend UI**
Open a **second** terminal and run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```
The application will automatically open in your web browser at `http://localhost:8501`.

---

## 🧪 Testing the Application (Test Cases)

I have included a few test CSV files in the `test_data` folder so you can see exactly how the application handles different patient scenarios. 

**How to test:**
Go to the **CSV Upload** tab in the web app, click "Browse Files", and upload the following files:

*   🟢 **`patient_stable.csv`**
    *   **What it is:** A completely healthy patient with normal vitals.
    *   **Expected Result:** The UI will show a green "PATIENT STABLE" banner. The confidence graphs will be completely green, proving the model knows when to stay quiet and avoid false alarms.

*   🔴 **`patient_sudden_crash.csv`**
    *   **What it is:** A patient who is stable for a few hours but then rapidly deteriorates (blood pressure drops, heart rate spikes).
    *   **Expected Result:** The UI will immediately flash a massive red "CRITICAL - FUTURE SHOCK EVENT DETECTED" banner. It will highlight exactly which rows the crash happens on, proving the model can catch a patient falling into danger.

*   🟡 **`patient_missing_data.csv`**
    *   **What it is:** A patient file where the sensors temporarily disconnected, leaving completely blank rows and missing data.
    *   **Expected Result:** The app won't crash! It demonstrates robust backend error-handling by automatically filling in the gaps (forward-filling previous vitals) and successfully predicting the risk score anyway.
