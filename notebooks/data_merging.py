import os

import pandas as pd


def merge_datasets():
    print("Loading datasets...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    path_small = os.path.join(data_dir, "diabetes.csv")
    path_large = os.path.join(data_dir, "diabetes_prediction_dataset.csv")

    df_small = pd.read_csv(path_small)
    df_large = pd.read_csv(path_large)

    print(f"Original diabetes.csv shape: {df_small.shape}")
    print(f"Original diabetes_prediction_dataset.csv shape: {df_large.shape}")

    if "Pregnancies" in df_small.columns:
        df_small = df_small.drop("Pregnancies", axis=1)

    rename_mapping = {
        "Glucose": "blood_glucose_level",
        "BloodPressure": "blood_pressure",
        "SkinThickness": "skin_thickness",
        "Insulin": "insulin",
        "BMI": "bmi",
        "DiabetesPedigreeFunction": "diabetes_pedigree_function",
        "Age": "age",
        "Outcome": "diabetes",
    }
    df_small = df_small.rename(columns=rename_mapping)
    df_small["gender"] = "Female"

    missing_in_small = [
        "hypertension",
        "heart_disease",
        "smoking_history",
        "HbA1c_level",
    ]
    for column in missing_in_small:
        df_small[column] = float("nan")

    missing_in_large = [
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "diabetes_pedigree_function",
    ]
    for column in missing_in_large:
        df_large[column] = float("nan")

    all_columns = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "diabetes_pedigree_function",
        "diabetes",
    ]

    df_small = df_small[all_columns]
    df_large = df_large[all_columns]
    df_merged = pd.concat([df_large, df_small], axis=0, ignore_index=True)

    print(f"Merged dataset shape: {df_merged.shape}")
    out_path = os.path.join(data_dir, "merged_diabetes.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"Successfully saved merged dataset to: {out_path}")


if __name__ == "__main__":
    merge_datasets()
