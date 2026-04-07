import pandas as pd
import numpy as np
import os

data_path = 'e:/my project GSG/classification/data/diabetes.csv'
cleaned_data_path = 'e:/my project GSG/classification/data/diabetes_cleaned.csv'

df = pd.read_csv(data_path)

# Columns where 0 makes no sense biologically (except Pregnancies, Outcome)
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

print("--- NULL VALUES AFTER REPLACING 0 WITH NaN ---")
print(df.isnull().sum())

# Fill NaNs with the mean of each column
for col in cols_to_replace:
    df[col] = df[col].fillna(df[col].mean())

print("\n--- NULL VALUES AFTER IMPUTATION ---")
print(df.isnull().sum())

print("\n--- DESCRIBE AFTER IMPUTATION ---")
print(df.describe())

# Save the cleaned dataset
df.to_csv(cleaned_data_path, index=False)
print(f"\nCleaned data saved to {cleaned_data_path}")
