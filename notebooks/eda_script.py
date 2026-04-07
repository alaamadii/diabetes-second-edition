import pandas as pd

df = pd.read_csv('e:/my project GSG/classification/data/diabetes.csv')
print("--- INFO ---")
df.info()

print("\n--- NULL VALUES ---")
print(df.isnull().sum())

print("\n--- ZERO VALUES IN COLUMNS ---")
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for c in cols_with_zeros:
    if c in df.columns:
        zeros = (df[c] == 0).sum()
        print(f"{c}: {zeros} zero values")

print("\n--- DESCRIBE ---")
print(df.describe())
