"""
process_mimic_data.py
=====================
ETL pipeline that transforms raw MIMIC-IV CSVs into a model-ready dataset.

Takes 3 input files (chartevents, icustays, patients) and produces a single
clean CSV with hourly patient records, engineered features, and binary shock
labels. The label uses a 2-hour lookahead to enable early warning.

7-step pipeline:
  1. Load raw CSVs from the MIMIC-IV export
  2. Filter chartevents to hemodynamic vital signs only
  3. Pivot from long format (one measurement per row) to wide (one row per hour)
  4. Consolidate arterial vs non-invasive BP into unified columns
  5. Join patient demographics (age, gender) from patients + icustays
  6. Engineer features (ShockIndex, temporal trends) and create shock labels
  7. Save final dataset to data/processed_mimic.csv

Usage:
  python ml_pipeline/process_mimic_data.py
"""

import pandas as pd
import numpy as np
import os
import sys

# --------------- CONFIGURATION ---------------

# Path to the raw MIMIC-IV CSV exports (downloaded from PhysioNet)
MIMIC_DATA_DIR = r"d:\Extra volume\mimic_4_data-20260411T161250Z-3-002\mimic_4_data"

# Output path
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_mimic.csv")

# MIMIC-IV chartevents uses numeric itemid codes for each measurement type.
# This mapping translates them to human-readable column names.
VITAL_ITEMS = {
    220045: "HeartRate",

    # Arterial (invasive) BP — preferred when available
    220050: "SysBP_Art",
    220051: "DiaBP_Art",
    220052: "MAP_Art",

    # Non-invasive BP — fallback when arterial line isn't placed
    220179: "SysBP_NI",
    220180: "DiaBP_NI",
    220181: "MAP_NI",

    220210: "RespRate",
    220277: "SpO2",
    223762: "Temperature",
    220074: "CVP",
}

VITAL_ITEMID_LIST = list(VITAL_ITEMS.keys())


def load_raw_data():
    """Load the 3 main MIMIC-IV CSV files."""
    print("=" * 60)
    print("  MIMIC-IV Data Processing Pipeline")
    print("=" * 60)

    print("\n[1/7] Loading raw CSV files...")

    charts_path = os.path.join(MIMIC_DATA_DIR, "chartevents.csv")
    icu_path = os.path.join(MIMIC_DATA_DIR, "icustays.csv")
    patients_path = os.path.join(MIMIC_DATA_DIR, "patients.csv")

    for p in [charts_path, icu_path, patients_path]:
        if not os.path.exists(p):
            print(f"  ERROR: File not found: {p}")
            sys.exit(1)

    # Load chartevents - this is the biggest file so we select only needed columns
    charts = pd.read_csv(
        charts_path,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        dtype={"subject_id": int, "hadm_id": int, "stay_id": int, "itemid": int},
        parse_dates=["charttime"],
    )
    print(f"  [OK] chartevents.csv  -> {len(charts):,} rows")

    icu = pd.read_csv(
        icu_path,
        parse_dates=["intime", "outtime"],
    )
    print(f"  [OK] icustays.csv     -> {len(icu):,} rows")

    patients = pd.read_csv(patients_path)
    print(f"  [OK] patients.csv     -> {len(patients):,} rows")

    return charts, icu, patients


def filter_vital_signs(charts):
    """Keep only rows whose itemid matches a vital sign we care about."""
    print("\n[2/7] Filtering for hemodynamic vital signs...")

    vitals = charts[charts["itemid"].isin(VITAL_ITEMID_LIST)].copy()

    # Drop rows where the numeric value is missing
    vitals = vitals.dropna(subset=["valuenum"])

    # Map itemid -> readable column name
    vitals["vital_name"] = vitals["itemid"].map(VITAL_ITEMS)

    print(f"  [OK] Kept {len(vitals):,} vital-sign measurements (from {len(charts):,} total)")
    print(f"  [OK] Unique stays with vitals: {vitals['stay_id'].nunique()}")

    # Show distribution
    print("\n  Vital sign counts:")
    for name, count in vitals["vital_name"].value_counts().items():
        print(f"    {name:20s}  {count:>8,}")

    return vitals


def create_hourly_pivot(vitals, icu):
    """
    Convert from long format (one measurement per row) to wide format
    (one row per stay_id + hour, with each vital as a separate column).
    """
    print("\n[3/7] Pivoting to hourly patient records...")

    # Merge with ICU stays to get the 'intime' reference point
    vitals = vitals.merge(
        icu[["stay_id", "intime"]],
        on="stay_id",
        how="inner",
    )

    # Calculate hours since ICU admission
    vitals["hours_in"] = (
        (vitals["charttime"] - vitals["intime"]).dt.total_seconds() / 3600
    ).round(0).astype(int)

    # Cap at 72 hours — beyond this, ICU stays get noisy and
    # the clinical picture changes significantly
    vitals = vitals[(vitals["hours_in"] >= 0) & (vitals["hours_in"] <= 72)]

    # Pivot: for each (stay_id, hour), average multiple readings.
    # ICU charting can produce several values per hour for the same vital.
    pivoted = vitals.pivot_table(
        index=["stay_id", "subject_id", "hours_in"],
        columns="vital_name",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()

    # Flatten column names
    pivoted.columns.name = None

    # Rename hours_in to Hour for clarity
    pivoted = pivoted.rename(columns={"hours_in": "Hour"})

    print(f"  [OK] Created {len(pivoted):,} hourly records")
    print(f"  [OK] Across {pivoted['stay_id'].nunique()} unique ICU stays")

    return pivoted


def consolidate_bp(df):
    """
    Merge arterial and non-invasive BP into a single set of columns.
    Prefer arterial (invasive) when available, fall back to non-invasive.
    """
    print("\n[4/7] Consolidating blood pressure columns...")

    # Systolic BP: prefer Arterial, fall back to Non-Invasive
    if "SysBP_Art" in df.columns and "SysBP_NI" in df.columns:
        df["SysBP"] = df["SysBP_Art"].fillna(df["SysBP_NI"])
    elif "SysBP_Art" in df.columns:
        df["SysBP"] = df["SysBP_Art"]
    elif "SysBP_NI" in df.columns:
        df["SysBP"] = df["SysBP_NI"]

    # Diastolic BP
    if "DiaBP_Art" in df.columns and "DiaBP_NI" in df.columns:
        df["DiaBP"] = df["DiaBP_Art"].fillna(df["DiaBP_NI"])
    elif "DiaBP_Art" in df.columns:
        df["DiaBP"] = df["DiaBP_Art"]
    elif "DiaBP_NI" in df.columns:
        df["DiaBP"] = df["DiaBP_NI"]

    # MAP (Mean Arterial Pressure)
    if "MAP_Art" in df.columns and "MAP_NI" in df.columns:
        df["MAP"] = df["MAP_Art"].fillna(df["MAP_NI"])
    elif "MAP_Art" in df.columns:
        df["MAP"] = df["MAP_Art"]
    elif "MAP_NI" in df.columns:
        df["MAP"] = df["MAP_NI"]

    # Drop the individual Art/NI columns
    drop_cols = [c for c in df.columns if c.endswith("_Art") or c.endswith("_NI")]
    df = df.drop(columns=drop_cols, errors="ignore")

    print(f"  [OK] Consolidated BP columns: SysBP, DiaBP, MAP")

    return df


def add_demographics(df, icu, patients):
    """Join patient age, gender, and ICU care unit info."""
    print("\n[5/7] Adding patient demographics...")

    # Get age and gender from patients
    demo = patients[["subject_id", "gender", "anchor_age", "dod"]].copy()
    demo = demo.rename(columns={"anchor_age": "Age"})

    # Encode gender as numeric (M=1, F=0)
    demo["Gender_M"] = (demo["gender"] == "M").astype(int)

    # Get care unit from icustays
    icu_info = icu[["stay_id", "first_careunit", "los"]].copy()

    # Merge demographics
    df = df.merge(demo[["subject_id", "Age", "Gender_M", "dod"]], on="subject_id", how="left")

    # Merge ICU info
    df = df.merge(icu_info, on="stay_id", how="left")

    print(f"  [OK] Added Age, Gender, Care Unit, LOS (Length of Stay)")
    print(f"  [OK] Age range: {df['Age'].min()} - {df['Age'].max()}")
    print(f"  [OK] Gender split: M={df['Gender_M'].sum():,} rows, F={(~df['Gender_M'].astype(bool)).sum():,} rows")

    return df


def engineer_features_and_label(df):
    """
    Create derived features and the binary shock label.

    Shock Label (rule-based clinical criteria):
      A patient-hour is labeled SHOCK (1) if ANY of these hold:
        - MAP < 65 mmHg        (hemodynamic instability)
        - Shock Index > 1.0    (HR/SysBP — tachycardia relative to pressure)
        - SysBP < 90 mmHg     (frank hypotension)
        - SpO2 < 90%           (hypoxia)
        - RespRate > 25 AND HeartRate > 110  (combined respiratory distress)

    The final label uses a 2-hour lookahead (shift -2) so the model
    predicts FUTURE shock, not current state — this is what makes it
    an early-warning system rather than a simple classifier.
    """
    print("\n[6/7] Engineering features and creating shock labels...")

    # -- Shock Index --
    df["ShockIndex"] = np.where(
        (df["SysBP"].notna()) & (df["SysBP"] > 0),
        np.round(df["HeartRate"] / df["SysBP"], 3),
        np.nan,
    )

    # -- Trend Features (change from previous hour, within same stay) --
    df = df.sort_values(["stay_id", "Hour"])

    df["HR_Change"] = df.groupby("stay_id")["HeartRate"].diff()
    df["SysBP_Change"] = df.groupby("stay_id")["SysBP"].diff()
    df["MAP_Change"] = df.groupby("stay_id")["MAP"].diff()
    df["SpO2_Change"] = df.groupby("stay_id")["SpO2"].diff()

    # -- SHOCK LABEL (Rule-Based) --
    df["Label_Shock"] = 0  # Default: Stable

    # Rule 1: MAP < 65
    if "MAP" in df.columns:
        df.loc[df["MAP"] < 65, "Label_Shock"] = 1

    # Rule 2: Shock Index > 1.0
    df.loc[df["ShockIndex"] > 1.0, "Label_Shock"] = 1

    # Rule 3: SysBP < 90
    if "SysBP" in df.columns:
        df.loc[df["SysBP"] < 90, "Label_Shock"] = 1
        
    # Rule 4: SpO2 < 90 (Hypoxia)
    if "SpO2" in df.columns:
        df.loc[df["SpO2"] < 90, "Label_Shock"] = 1
        
    # Rule 5: Severe Respiratory / Tachycardia Distress
    if "RespRate" in df.columns and "HeartRate" in df.columns:
        df.loc[(df["RespRate"] > 25) & (df["HeartRate"] > 110), "Label_Shock"] = 1

    # 2-hour lookahead: predict whether the patient will be in shock
    # 2 hours from now. This prevents target leakage and creates a
    # genuinely predictive (not reactive) early-warning system.
    df["Label_Shock_Next"] = df.groupby("stay_id")["Label_Shock"].shift(-2)

    # End-of-stay rows have no future — assume they stay in current state
    df["Label_Shock_Next"] = df["Label_Shock_Next"].fillna(df["Label_Shock"]).astype(int)

    # Count labels
    shock_count = df["Label_Shock_Next"].sum()
    stable_count = len(df) - shock_count
    print(f"  [OK] Shock Index computed")
    print(f"  [OK] Trend features added (HR_Change, SysBP_Change, MAP_Change, SpO2_Change)")
    print(f"\n  -- Label Distribution --")
    print(f"    Stable (0): {stable_count:,} rows ({stable_count/len(df)*100:.1f}%)")
    print(f"    Shock  (1): {shock_count:,} rows ({shock_count/len(df)*100:.1f}%)")

    return df


def save_dataset(df):
    """Save the final processed dataset."""
    print("\n[7/7] Saving processed dataset...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select the final columns for model training
    feature_cols = [
        "stay_id", "subject_id", "Hour",
        "HeartRate", "SysBP", "DiaBP", "MAP", "RespRate", "SpO2",
        "Temperature", "CVP",
        "ShockIndex", "HR_Change", "SysBP_Change", "MAP_Change", "SpO2_Change",
        "Age", "Gender_M",
        "Label_Shock", "Label_Shock_Next",
    ]

    # Only keep columns that actually exist
    final_cols = [c for c in feature_cols if c in df.columns]
    df_final = df[final_cols].copy()

    # Drop rows where critical vitals are ALL missing
    critical = ["HeartRate", "SysBP", "MAP", "RespRate", "SpO2"]
    critical_present = [c for c in critical if c in df_final.columns]
    df_final = df_final.dropna(subset=critical_present, how="all")

    # Forward-fill small gaps (max 2 hours) within each stay.
    # This handles brief sensor disconnections without hallucinating
    # vitals hours after a real data gap.
    fill_cols = ["HeartRate", "SysBP", "DiaBP", "MAP", "RespRate", "SpO2", "Temperature", "CVP"]
    fill_present = [c for c in fill_cols if c in df_final.columns]
    df_final[fill_present] = df_final.groupby("stay_id")[fill_present].ffill(limit=2)

    df_final.to_csv(OUTPUT_FILE, index=False)

    print(f"  [OK] Saved to: {OUTPUT_FILE}")
    print(f"  [OK] Final shape: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")
    print(f"  [OK] Unique patients: {df_final['subject_id'].nunique()}")
    print(f"  [OK] Unique ICU stays: {df_final['stay_id'].nunique()}")

    # Preview
    print("\n  -- Sample Data (first 5 rows) --")
    print(df_final.head().to_string(index=False))

    return df_final


def main():
    # Step 1: Load
    charts, icu, patients = load_raw_data()

    # Step 2: Filter
    vitals = filter_vital_signs(charts)

    # Step 3: Pivot
    pivoted = create_hourly_pivot(vitals, icu)

    # Step 4: Consolidate BP
    pivoted = consolidate_bp(pivoted)

    # Step 5: Demographics
    pivoted = add_demographics(pivoted, icu, patients)

    # Step 6: Features + Labels
    pivoted = engineer_features_and_label(pivoted)

    # Step 7: Save
    final_df = save_dataset(pivoted)

    print("\n" + "=" * 60)
    print("  DONE! Pipeline complete. Dataset ready for model training.")
    print("=" * 60)

    return final_df


if __name__ == "__main__":
    main()
