import numpy as np
import pandas as pd

YES_NO_COLS = [
    "Appetite_Loss",
    "Vomiting",
    "Diarrhea",
    "Coughing",
    "Labored_Breathing",
    "Lameness",
    "Skin_Lesions",
    "Nasal_Discharge",
    "Eye_Discharge",
]


def categorize_temperature(value):
    if pd.isna(value):
        return "Unknown"
    if value < 39.0:
        return "Low"
    if value <= 39.5:
        return "Normal"
    return "High"


def categorize_heart_rate(value):
    if pd.isna(value):
        return "Unknown"
    if value < 80:
        return "Low"
    if value <= 130:
        return "Normal"
    return "High"


def calculate_severity_level(row):
    symptom_score = sum(str(row.get(col, "")).strip().lower() == "yes" for col in YES_NO_COLS)
    duration_days = row.get("Duration_Days")
    temperature_category = row.get("Temperature_Category", "Unknown")
    heart_rate_category = row.get("Heart_Rate_Category", "Unknown")

    score = symptom_score

    if pd.notna(duration_days):
        if duration_days >= 7:
            score += 2
        elif duration_days >= 4:
            score += 1

    if temperature_category in {"Low", "High"}:
        score += 1

    if heart_rate_category in {"Low", "High"}:
        score += 1

    if score <= 3:
        return "Mild"
    if score <= 6:
        return "Moderate"
    return "Severe"


def add_engineered_features(df):
    engineered_df = df.copy()
    engineered_df["Temperature_Category"] = engineered_df["Body_Temperature"].apply(categorize_temperature)
    engineered_df["Heart_Rate_Category"] = engineered_df["Heart_Rate"].apply(categorize_heart_rate)
    engineered_df["Severity_Level"] = engineered_df.apply(calculate_severity_level, axis=1)
    return engineered_df


def build_correlation_frame(df):
    correlation_df = df.copy()
    category_maps = {
        "Temperature_Category": {"Low": 0, "Normal": 1, "High": 2, "Unknown": np.nan},
        "Heart_Rate_Category": {"Low": 0, "Normal": 1, "High": 2, "Unknown": np.nan},
        "Severity_Level": {"Mild": 1, "Moderate": 2, "Severe": 3},
    }

    for column, mapping in category_maps.items():
        if column in correlation_df.columns:
            correlation_df[column] = correlation_df[column].map(mapping)

    return correlation_df
