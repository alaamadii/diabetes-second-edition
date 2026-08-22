import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def prepare_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    target_col = "diabetes"
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    categorical_cols = ["gender", "smoking_history"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)

    print(f"--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return model


def main():
    data_file = "data/merged_diabetes.csv"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return

    pos_cases = y_train.sum()
    neg_cases = len(y_train) - pos_cases
    scale_pos = neg_cases / pos_cases if pos_cases > 0 else 1.0
    print(f"Applying scale_pos_weight = {scale_pos:.2f} for XGBoost...")

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos,
                ),
            ),
        ]
    )
    xgb_pipeline = train_and_evaluate(
        xgb_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        "XGBoost classifier",
    )
    joblib.dump(xgb_pipeline, os.path.join(models_dir, "xgboost.pkl"))
    print("XGBoost model trained and saved successfully.")


if __name__ == "__main__":
    main()
