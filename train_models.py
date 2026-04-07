import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = 'diabetes'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    categorical_cols = ['gender', 'smoking_history']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)
    
    print(f"--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def main():
    data_file = 'data/merged_diabetes.csv'
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        
    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return

    # Calculate pos weight for XGBoost to balance classes and prioritize Recall
    pos_cases = y_train.sum()
    neg_cases = len(y_train) - pos_cases
    scale_pos = neg_cases / pos_cases if pos_cases > 0 else 1.0
    print(f"Applying scale_pos_weight = {scale_pos:.2f} for XGBoost...")

    # 1. XGBoost
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos))])
    xgb_pipeline = train_and_evaluate(xgb_pipeline, X_train, y_train, X_test, y_test, "XGBoost (High-Recall Architecture)")
    joblib.dump(xgb_pipeline, os.path.join(models_dir, 'xgboost.pkl'))
    
    print("XGBoost model trained and saved successfully.")

if __name__ == "__main__":
    main()
