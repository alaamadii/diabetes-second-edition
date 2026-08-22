import os

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.modeling import (
    build_preprocessor,
    compute_scale_pos_weight,
    evaluate_binary_classifier,
    split_features_target,
)


def prepare_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_features_target(df)
    preprocessor = build_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)

    print(f"--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_binary_classifier(y_test, y_pred, y_score)
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"Specificity: {metrics.specificity:.4f}")
    if metrics.roc_auc is not None:
        print(f"ROC-AUC: {metrics.roc_auc:.4f}")
    if metrics.pr_auc is not None:
        print(f"PR-AUC: {metrics.pr_auc:.4f}")

    return model, metrics


def main():
    data_file = "data/merged_diabetes.csv"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    try:
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(data_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not prepare data: {exc}")
        return

    scale_pos = compute_scale_pos_weight(y_train)
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
    xgb_pipeline, _ = train_and_evaluate(
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
