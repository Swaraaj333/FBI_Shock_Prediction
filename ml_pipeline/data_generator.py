import pandas as pd
import numpy as np
import os

def generate_patient_data(status="stable", num_hours=24):
    """
    Generate synthetic time-series vitals simulating MIMIC-IV formatted data.
    status: 'stable' or 'shock'
    num_hours: total hours to simulate (default 24)
    """
    
    # Establish base vitals
    time_points = range(num_hours)
    
    # 1. Healthy baselines
    heart_rate = np.random.normal(75, 5, num_hours)  # Normal HR: ~75
    sys_bp = np.random.normal(120, 5, num_hours)     # Normal SysBP: ~120
    resp_rate = np.random.normal(16, 2, num_hours)   # Normal RR: ~16
    spo2 = np.random.normal(98, 1, num_hours)        # Normal SpO2: ~98
    
    if status == "shock":
        # Simulate deteriorating vitals starting halfway through the timeframe
        deterioration_start = num_hours // 2
        deterioration_steps = num_hours - deterioration_start
        
        # Gradually increase HR by up to 40 bpm
        hr_increase = np.linspace(0, 40, deterioration_steps)
        heart_rate[deterioration_start:] += hr_increase + np.random.normal(0, 3, deterioration_steps)
        
        # Gradually decrease Systolic BP by up to 50 mmHg
        bp_decrease = np.linspace(0, 50, deterioration_steps)
        sys_bp[deterioration_start:] -= bp_decrease + np.random.normal(0, 3, deterioration_steps)
        
        # Gradually increase Respiratory Rate
        rr_increase = np.linspace(0, 10, deterioration_steps)
        resp_rate[deterioration_start:] += rr_increase
        
        # Gradually decrease SpO2
        spo2_decrease = np.linspace(0, 5, deterioration_steps)
        spo2[deterioration_start:] -= spo2_decrease

    # Assemble the dataframe
    df = pd.DataFrame({
        "Hour": time_points,
        "HeartRate": np.round(heart_rate, 1),
        "SysBP": np.round(sys_bp, 1),
        "RespRate": np.round(resp_rate, 1),
        "SpO2": np.round(spo2, 1)
    })
    
    # Optional explicitly derived metric to display for the user
    # Shock Index = HR / SysBP. A normal value is 0.5 - 0.7. Values > 0.9 indicate shock.
    df["ShockIndex"] = np.round(df["HeartRate"] / df["SysBP"], 2)
    
    # Add a target label (1 = Shock, 0 = Stable) for our ML training script later
    df["Label_Shock"] = 1 if status == "shock" else 0
    
    return df

if __name__ == "__main__":
    print("Generating simulated MIMIC-IV time-series patient data...")
    
    # Generate the samples
    stable_df = generate_patient_data("stable")
    shock_df = generate_patient_data("shock")
    
    # Make sure an output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    stable_df.to_csv("data/stable_patient.csv", index=False)
    shock_df.to_csv("data/shock_patient.csv", index=False)
    
    print("Files successfully saved to the 'data/' folder!")
    print("- data/stable_patient.csv")
    print("- data/shock_patient.csv")
