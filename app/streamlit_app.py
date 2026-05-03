"""
streamlit_app.py
================
Streamlit frontend for the Hemodynamic Shock Prediction System.

This UI communicates with the Flask API (flask_api.py) running on port 5000.
It does NOT load the ML model directly — all predictions flow through the API
to maintain a clean separation between the presentation and inference layers.

Three tabs:
  1. Manual Entry  — Sliders for real-time single-patient assessment
  2. CSV Upload    — Batch timeline analysis with risk banners
  3. Data Explorer — Interactive MIMIC-IV dataset visualizations

Usage:
  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import joblib
import traceback

# ── Page Config ──────────────────────────────────────────

st.set_page_config(
    page_title="Hemodynamic Shock Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
# Glassmorphism dark theme with gradient backgrounds, pulsing
# danger animations, and styled metric cards for a premium feel.

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Header styling */
    h1, h2, h3 {
        color: #e0e0ff !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    [data-testid="stMetricLabel"] {
        color: #a0a0cc !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    }

    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        backdrop-filter: blur(12px);
    }

    /* Risk badge */
    .risk-high {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
    }

    .risk-low {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
    }

    /* Slider styling */
    .stSlider > div > div > div {
        color: #a0a0cc;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: #a0a0cc;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Model Logic ──────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

@st.cache_resource
def load_ml_model():
    """Load the serialized model and feature list from disk."""
    model_path = os.path.join(MODEL_DIR, "shock_rf_model.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if not all(os.path.exists(p) for p in [model_path, features_path]):
        return None, None

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return model, feature_names

# Load model at startup
MODEL, FEATURE_NAMES = load_ml_model()

def prepare_input(data_dict):
    """Normalize and enrich a single row of vitals for model input."""
    row = {}
    field_map = {
        "HeartRate": "HeartRate", "heart_rate": "HeartRate", "hr": "HeartRate",
        "SysBP": "SysBP", "sys_bp": "SysBP", "systolic_bp": "SysBP",
        "MAP": "MAP", "map": "MAP", "mean_arterial_pressure": "MAP",
        "RespRate": "RespRate", "resp_rate": "RespRate", "respiratory_rate": "RespRate",
        "SpO2": "SpO2", "spo2": "SpO2", "oxygen_saturation": "SpO2",
        "Age": "Age", "age": "Age", "Gender_M": "Gender_M", "gender_m": "Gender_M",
        "ShockIndex": "ShockIndex", "shock_index": "ShockIndex",
        "HR_Change": "HR_Change", "SysBP_Change": "SysBP_Change",
        "MAP_Change": "MAP_Change", "SpO2_Change": "SpO2_Change",
    }

    for key, val in data_dict.items():
        mapped = field_map.get(key, key)
        if mapped in FEATURE_NAMES:
            try:
                row[mapped] = float(val) if val is not None and val != "" else np.nan
            except (ValueError, TypeError):
                row[mapped] = np.nan

    if "ShockIndex" not in row or np.isnan(row.get("ShockIndex", np.nan)):
        hr, sbp = row.get("HeartRate", np.nan), row.get("SysBP", np.nan)
        if not np.isnan(hr) and not np.isnan(sbp) and sbp > 0:
            row["ShockIndex"] = round(hr / sbp, 3)

    for feat in FEATURE_NAMES:
        if feat not in row: row[feat] = np.nan
    return row

def predict_vitals(vitals_dict):
    """Run inference locally."""
    if MODEL is None: return {"error": "Model not loaded"}
    try:
        prepared = prepare_input(vitals_dict)
        df = pd.DataFrame([prepared])[FEATURE_NAMES]
        prediction = int(MODEL.predict(df)[0])
        probabilities = MODEL.predict_proba(df)[0]

        return {
            "prediction": prediction,
            "label": "SHOCK" if prediction == 1 else "STABLE",
            "confidence": round(float(max(probabilities)) * 100, 1),
            "prob_stable": round(float(probabilities[0]) * 100, 1),
            "prob_shock": round(float(probabilities[1]) * 100, 1),
            "shock_index": prepared.get("ShockIndex"),
        }
    except Exception as e:
        return {"error": str(e)}

def predict_csv(file):
    """Process a CSV file locally with full temporal feature engineering."""
    if MODEL is None: return {"error": "Model not loaded"}
    try:
        df = pd.read_csv(file)
        vitals_cols = [c for c in df.columns if c not in ['stay_id', 'subject_id', 'charttime']]
        
        # Temporal Preprocessing (Forward-fill and Change Features)
        if "stay_id" in df.columns:
            df[vitals_cols] = df.groupby("stay_id")[vitals_cols].ffill(limit=2)
        else:
            df[vitals_cols] = df[vitals_cols].ffill(limit=2)

        if 'HeartRate' in df.columns and 'SysBP' in df.columns:
            df['ShockIndex'] = np.where((df['SysBP'].notna()) & (df['SysBP'] > 0), 
                                       np.round(df['HeartRate'] / df['SysBP'], 3), np.nan)

        change_map = {'HeartRate': 'HR_Change', 'SysBP': 'SysBP_Change', 'MAP': 'MAP_Change', 'SpO2': 'SpO2_Change'}
        for col, change_col in change_map.items():
            if col in df.columns:
                df[change_col] = df.groupby('stay_id')[col].diff() if 'stay_id' in df.columns else df[col].diff()

        results = []
        for idx, row in df.iterrows():
            pred = predict_vitals(row.to_dict())
            pred["row_index"] = idx
            results.append(pred)

        n_shock = sum(1 for r in results if r["prediction"] == 1)
        return {
            "total_rows": len(results),
            "shock_count": n_shock,
            "stable_count": len(results) - n_shock,
            "shock_percentage": round(n_shock / len(results) * 100, 1) if results else 0,
            "patient_status": "CRITICAL_RISK" if n_shock > 0 else "STABLE",
            "predictions": results,
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


def create_gauge(value, title, min_val, max_val, danger_threshold, danger_dir="above"):  # noqa: E501
    """Plotly gauge — turns red when a vital breaches its clinical threshold."""
    if danger_dir == "above":
        color = "#ff4444" if value > danger_threshold else "#00c851"
    else:
        color = "#ff4444" if value < danger_threshold else "#00c851"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#a0a0cc"}},
        number={"font": {"size": 28, "color": "white"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#555"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, max_val], "color": "rgba(255,255,255,0.03)"},
            ],
        },
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0a0cc"},
    )
    return fig


def create_risk_donut(prob_stable, prob_shock):
    """Donut chart — green/red split showing model confidence at a glance."""
    colors = ["#00c851", "#ff4444"]
    fig = go.Figure(data=[go.Pie(
        labels=["Stable", "Shock"],
        values=[prob_stable, prob_shock],
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0.3)", width=2)),
        textinfo="percent",
        textfont=dict(size=14, color="white"),
    )])

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0a0cc"},
        showlegend=True,
        legend=dict(
            font=dict(color="#a0a0cc", size=12),
            orientation="h",
            yanchor="bottom", y=-0.1, xanchor="center", x=0.5,
        ),
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Shock Predictor")
    st.markdown("---")

    # Model Status
    if MODEL is not None:
        st.success("ML Model Loaded")
    else:
        st.error("ML Model Missing")

    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This system predicts **future hemodynamic shock risk** 
    (1-2 hours in advance) using real **MIMIC-IV** ICU data.
    
    **Model:** Random Forest  
    **Accuracy:** ~78.0% (Predictive)  
    **Data:** 6,685 hourly records  
    **Patients:** 100 ICU patients
    """)

    st.markdown("---")
    st.markdown("### Clinical Thresholds")
    st.markdown("""
    - **MAP < 65 mmHg** = Shock
    - **Shock Index > 1.0** = Shock  
    - **SysBP < 90 mmHg** = Shock
    """)


# ── Main Content ─────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 style="font-size: 2.5rem; background: linear-gradient(135deg, #667eea, #764ba2, #f093fb); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 5px;">
    Future Hemodynamic Shock Predictor</h1>
    <p style="color: #8888bb; font-size: 1.1rem;">
    Early-warning ICU patient monitoring powered by MIMIC-IV data & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["Manual Entry", "CSV Upload", "Data Explorer"])

# ── Tab 1: Manual Entry ──────────────────────────────────
# Nurse-facing "pocket calculator" — input vitals via sliders,
# get an instant risk assessment with gauge charts.

with tab1:
    st.markdown("### Enter Patient Vitals")
    st.markdown("Adjust the sliders to input current patient measurements.")

    col1, col2, col3 = st.columns(3)

    with col1:
        heart_rate = st.slider("Heart Rate (bpm)", 30, 200, 80, key="hr")
        resp_rate = st.slider("Respiratory Rate (breaths/min)", 5, 45, 16, key="rr")
        age = st.slider("Patient Age", 18, 100, 55, key="age")

    with col2:
        sys_bp = st.slider("Systolic BP (mmHg)", 50, 200, 120, key="sbp")
        spo2 = st.slider("SpO2 (%)", 70, 100, 97, key="spo2")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")

    with col3:
        map_val = st.slider("MAP (mmHg)", 30, 130, 75, key="map")
        temperature = st.slider("Temperature (C)", 34.0, 42.0, 37.0, 0.1, key="temp")
        cvp = st.slider("CVP (mmHg)", 0, 30, 8, key="cvp")

    # Calculate and display Shock Index
    shock_index = round(heart_rate / sys_bp, 3) if sys_bp > 0 else 0
    si_color = "red" if shock_index > 1.0 else ("orange" if shock_index > 0.7 else "green")

    st.markdown("---")

    # Predict button
    col_btn, col_si = st.columns([1, 1])
    with col_btn:
        predict_btn = st.button("Predict Shock Risk", use_container_width=True, type="primary")
    with col_si:
        st.metric("Shock Index (HR/SysBP)", f"{shock_index:.3f}",
                  delta="DANGER" if shock_index > 1.0 else "Normal",
                  delta_color="inverse" if shock_index > 1.0 else "normal")

    if predict_btn:
        if MODEL is None:
            st.error("ML Model is not loaded! Check 'model/' folder.")
        else:
            vitals = {
                "HeartRate": heart_rate,
                "SysBP": sys_bp,
                "MAP": map_val,
                "RespRate": resp_rate,
                "SpO2": spo2,
                "Age": age,
                "Gender_M": 1 if gender == "Male" else 0,
                "ShockIndex": shock_index,
                "HR_Change": 0,
                "SysBP_Change": 0,
                "MAP_Change": 0,
                "SpO2_Change": 0,
            }

            with st.spinner("Analyzing vitals..."):
                time.sleep(0.5)  # Brief pause for visual effect
                result = predict_vitals(vitals)

            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                st.markdown("---")

                # Risk Banner
                if result["prediction"] == 1:
                    st.markdown(
                        '<div class="risk-high">SHOCK RISK DETECTED - IMMEDIATE ATTENTION REQUIRED</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="risk-low">PATIENT STABLE - Continue Monitoring</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Results row
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("Prediction", result["label"])
                with res_col2:
                    st.metric("Confidence", f"{result['confidence']}%")
                with res_col3:
                    st.metric("Shock Probability", f"{result['prob_shock']}%")
                with res_col4:
                    st.metric("Stable Probability", f"{result['prob_stable']}%")

                # Charts row
                st.markdown("<br>", unsafe_allow_html=True)
                ch_col1, ch_col2, ch_col3 = st.columns([2, 1, 2])

                with ch_col1:
                    st.markdown("##### Vital Signs Gauges")
                    g1, g2 = st.columns(2)
                    with g1:
                        st.plotly_chart(
                            create_gauge(heart_rate, "Heart Rate", 30, 200, 100, "above"),
                            use_container_width=True,
                        )
                        st.plotly_chart(
                            create_gauge(resp_rate, "Resp Rate", 5, 45, 25, "above"),
                            use_container_width=True,
                        )
                    with g2:
                        st.plotly_chart(
                            create_gauge(sys_bp, "Systolic BP", 50, 200, 90, "below"),
                            use_container_width=True,
                        )
                        st.plotly_chart(
                            create_gauge(spo2, "SpO2", 70, 100, 93, "below"),
                            use_container_width=True,
                        )

                with ch_col2:
                    st.markdown("##### Risk Distribution")
                    st.plotly_chart(
                        create_risk_donut(result["prob_stable"], result["prob_shock"]),
                        use_container_width=True,
                    )

                with ch_col3:
                    st.markdown("##### Clinical Assessment")
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

                    checks = [
                        ("MAP >= 65 mmHg", map_val >= 65),
                        ("Systolic BP >= 90 mmHg", sys_bp >= 90),
                        ("Shock Index <= 1.0", shock_index <= 1.0),
                        ("Heart Rate 60-100 bpm", 60 <= heart_rate <= 100),
                        ("SpO2 >= 94%", spo2 >= 94),
                        ("Resp Rate 12-20", 12 <= resp_rate <= 20),
                        ("Temperature 36.5-37.5 C", 36.5 <= temperature <= 37.5),
                    ]

                    for label, ok in checks:
                        icon = "[PASS]" if ok else "[FAIL]"
                        color = "#00c851" if ok else "#ff4444"
                        st.markdown(
                            f'<p style="color:{color}; margin:6px 0; font-size:0.95rem;">'
                            f'{icon} {label}</p>',
                            unsafe_allow_html=True,
                        )

                    st.markdown('</div>', unsafe_allow_html=True)


# ── Tab 2: CSV Upload ────────────────────────────────────
# Upload a patient timeline CSV. The API handles forward-filling,
# temporal feature engineering, and per-row prediction.

with tab2:
    st.markdown("### Upload Patient CSV")
    st.markdown("""
    Upload a CSV file containing patient vital signs. 
    The system will predict shock risk for each row.
    
    **Required columns:** `HeartRate`, `SysBP`, `MAP`, `RespRate`, `SpO2`  
    **Optional columns:** `Age`, `Gender_M`, `ShockIndex`
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV with patient vital signs data",
    )

    if uploaded_file is not None:
        # Preview
        preview_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset for API upload

        st.markdown(f"**Preview** ({len(preview_df)} rows)")
        st.dataframe(preview_df.head(10), use_container_width=True)

        if st.button("Run Predictions", type="primary", key="csv_predict"):
            if MODEL is None:
                st.error("ML Model is not loaded! Check 'model/' folder.")
            else:
                with st.spinner(f"Processing {len(preview_df)} rows..."):
                    result = predict_csv(uploaded_file)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("---")

                    # Summary metrics
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    with sm1:
                        st.metric("Total Rows", result["total_rows"])
                    with sm2:
                        st.metric("Stable", result["stable_count"])
                    with sm3:
                        st.metric("Shock", result["shock_count"])
                    with sm4:
                        st.metric("Shock %", f"{result['shock_percentage']}%")

                    # Results table
                    pred_df = pd.DataFrame(result["predictions"])
                    display_cols = ["row_index", "label", "confidence", "prob_shock", "prob_stable"]
                    avail_cols = [c for c in display_cols if c in pred_df.columns]

                    st.markdown("#### Prediction Results")
                    st.dataframe(
                        pred_df[avail_cols].style.format(
                            {"confidence": "{:.1f}%", "prob_shock": "{:.1f}%", "prob_stable": "{:.1f}%"}
                        ).apply(
                            lambda x: [
                                "background-color: rgba(255,68,68,0.2)" if v == "SHOCK" else ""
                                for v in x
                            ],
                            subset=["label"],
                        ),
                        use_container_width=True,
                    )

                    # Patient Status Alert
                    if result.get("patient_status") == "CRITICAL_RISK":
                        st.markdown(
                            '<div class="risk-high">CRITICAL - FUTURE SHOCK EVENT DETECTED IN TIMELINE</div>',
                            unsafe_allow_html=True,
                        )
                        shock_rows = pred_df[pred_df["label"] == "SHOCK"]["row_index"].tolist()
                        st.warning(f"Patient is predicted to enter Hemodynamic Shock at rows: {shock_rows}")
                    else:
                        st.markdown(
                            '<div class="risk-low">PATIENT STABLE - No Shock Events Predicted</div>',
                            unsafe_allow_html=True,
                        )

                    # Confidence histogram
                    fig2 = px.histogram(
                        pred_df, x="confidence",
                        color="label",
                        color_discrete_map={"STABLE": "#00c851", "SHOCK": "#ff4444"},
                        nbins=20,
                        title="Prediction Confidence Distribution",
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#a0a0cc"),
                        height=300,
                    )
                    st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: Data Explorer ─────────────────────────────────
# Interactive visualizations of the processed MIMIC-IV training data.
# Useful for understanding label distributions, vital distributions,
# feature correlations, and individual patient timelines.

with tab3:
    st.markdown("### Explore Processed MIMIC-IV Data")

    # This path is relative to where streamlit is invoked from (project root)
    data_path = "data/processed_mimic.csv"
    try:
        df = pd.read_csv(data_path)

        # Summary
        st.markdown(f"**Dataset:** {len(df):,} rows | {df['stay_id'].nunique()} ICU stays | {df['subject_id'].nunique()} patients")

        ex1, ex2 = st.columns(2)

        with ex1:
            st.markdown("#### Label Distribution (Future Risk)")
            label_counts = df["Label_Shock_Next"].value_counts()
            fig = px.bar(
                x=["Stable", "Future Shock"],
                y=[label_counts.get(0, 0), label_counts.get(1, 0)],
                color=["Stable", "Future Shock"],
                color_discrete_map={"Stable": "#00c851", "Future Shock": "#ff4444"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0a0cc"),
                showlegend=False,
                height=300,
                xaxis_title="", yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)

        with ex2:
            st.markdown("#### Vital Signs Distribution")
            vital_choice = st.selectbox(
                "Select Vital Sign",
                ["HeartRate", "SysBP", "MAP", "RespRate", "SpO2", "ShockIndex"],
            )
            if vital_choice in df.columns:
                fig = px.histogram(
                    df, x=vital_choice, color="Label_Shock_Next",
                    color_discrete_map={0: "#00c851", 1: "#ff4444"},
                    nbins=40, barmode="overlay", opacity=0.7,
                    labels={"Label_Shock_Next": "Future Shock Risk"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#a0a0cc"),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### Feature Correlations (against Future Risk)")
        numeric_cols = ["HeartRate", "SysBP", "MAP", "RespRate", "SpO2", "ShockIndex", "Age", "Label_Shock_Next"]
        avail_numeric = [c for c in numeric_cols if c in df.columns]
        corr = df[avail_numeric].corr()

        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a0a0cc"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Patient timeline
        st.markdown("#### Patient Timeline")
        stay_ids = sorted(df["stay_id"].unique())
        selected_stay = st.selectbox("Select ICU Stay", stay_ids)

        stay_data = df[df["stay_id"] == selected_stay].sort_values("Hour")

        if len(stay_data) > 0:
            vitals_to_plot = ["HeartRate", "SysBP", "MAP", "SpO2"]
            avail_vitals = [v for v in vitals_to_plot if v in stay_data.columns]

            fig = go.Figure()
            colors = {"HeartRate": "#667eea", "SysBP": "#f093fb", "MAP": "#ff6b6b", "SpO2": "#00c851"}

            for vital in avail_vitals:
                fig.add_trace(go.Scatter(
                    x=stay_data["Hour"],
                    y=stay_data[vital],
                    mode="lines+markers",
                    name=vital,
                    line=dict(color=colors.get(vital, "#ffffff"), width=2),
                    marker=dict(size=4),
                ))

            # Shade shock hours
            shock_hours = stay_data[stay_data["Label_Shock_Next"] == 1]["Hour"]
            for h in shock_hours:
                fig.add_vrect(x0=h-0.5, x1=h+0.5, fillcolor="rgba(255,68,68,0.1)", line_width=0)

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0a0cc"),
                height=400,
                xaxis_title="Hours Since ICU Admission",
                yaxis_title="Value",
                legend=dict(font=dict(color="#a0a0cc")),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Red shaded areas indicate hours where the patient was in shock state.")

        # Raw data view
        with st.expander("View Raw Data"):
            st.dataframe(df.head(100), use_container_width=True)

    except FileNotFoundError:
        st.warning("Processed data not found. Run process_mimic_data.py first.")


# ── Footer ───────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p>Hemodynamic Shock Prediction System | Built with MIMIC-IV Data | 
    Random Forest ML Model | Streamlit + Flask Architecture</p>
</div>
""", unsafe_allow_html=True)
