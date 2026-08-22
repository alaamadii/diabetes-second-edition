from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "diabetes"
CATEGORICAL_COLUMNS = ["gender", "smoking_history"]
FEATURE_COLUMNS = [
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
]


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    roc_auc: float | None
    pr_auc: float | None


def validate_schema(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def split_features_target(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    validate_schema(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_columns = [
        column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
    ]

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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    positive_cases = float(y_train.sum())
    negative_cases = float(len(y_train) - positive_cases)
    if positive_cases <= 0:
        return 1.0
    return negative_cases / positive_cases


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_score=None,
) -> EvaluationMetrics:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    true_negative = int(((y_true_array == 0) & (y_pred_array == 0)).sum())
    false_positive = int(((y_true_array == 0) & (y_pred_array == 1)).sum())
    denominator = true_negative + false_positive
    specificity = true_negative / denominator if denominator else 0.0

    roc_auc = None
    pr_auc = None
    if y_score is not None and len(np.unique(y_true_array)) == 2:
        roc_auc = float(roc_auc_score(y_true_array, y_score))
        pr_auc = float(average_precision_score(y_true_array, y_score))

    return EvaluationMetrics(
        accuracy=float(accuracy_score(y_true_array, y_pred_array)),
        precision=float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        recall=float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        specificity=float(specificity),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )
