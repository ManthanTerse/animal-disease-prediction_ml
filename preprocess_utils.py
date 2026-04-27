import pandas as pd
import numpy as np
import re

# Convert values like "102 F" -> 102, "3 days" -> 3
def extract_number(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    number = re.findall(r"\d+\.?\d*", value)

    if number:
        return float(number[0])
    else:
        return np.nan


# Convert different yes/no forms to standard values
def clean_yes_no(value):
    value = str(value).strip().lower()
    if value in ["yes", "y", "true", "1"]:
        return "Yes"
    if value in ["no", "n", "false", "0"]:
        return "No"
    return ""


# Standardize text (cow -> Cow, german shepherd -> German Shepherd)
def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().title()


# Main preprocessing function
def preprocess_df(df):
    df = df.copy()

    # Convert duration to number
    if "Duration" in df.columns:
        df["Duration_Days"] = df["Duration"].apply(extract_number)
        df.drop("Duration", axis=1, inplace=True)

    # Convert numeric text columns
    numeric_cols = ["Body_Temperature", "Heart_Rate"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(extract_number)

    # Convert Yes/No columns
    yes_no_cols = [
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

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_yes_no)

    # Clean text columns
    text_cols = [
        "Animal_Type",
        "Breed",
        "Gender",
        "Symptom_1",
        "Symptom_2",
        "Symptom_3",
        "Symptom_4",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    return df