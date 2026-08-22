import unittest

import numpy as np
import pandas as pd

from src.modeling import (
    FEATURE_COLUMNS,
    build_preprocessor,
    compute_scale_pos_weight,
    evaluate_binary_classifier,
    split_features_target,
    validate_schema,
)


class ModelingTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for index in range(20):
            rows.append(
                {
                    "gender": "Female" if index % 2 == 0 else "Male",
                    "age": 20 + index,
                    "hypertension": index % 2,
                    "heart_disease": (index // 2) % 2,
                    "smoking_history": "never" if index % 3 else "former",
                    "bmi": 22.0 + index / 10,
                    "HbA1c_level": 5.0 + index / 20,
                    "blood_glucose_level": 90 + index,
                    "blood_pressure": np.nan if index % 4 == 0 else 70 + index,
                    "skin_thickness": 20 + index,
                    "insulin": np.nan if index % 5 == 0 else 80 + index,
                    "diabetes_pedigree_function": 0.3 + index / 100,
                    "diabetes": index % 2,
                }
            )
        self.df = pd.DataFrame(rows)

    def test_validate_schema_rejects_missing_feature(self):
        invalid = self.df.drop(columns=[FEATURE_COLUMNS[0]])
        with self.assertRaises(ValueError):
            validate_schema(invalid)

    def test_split_is_stratified_and_reproducible(self):
        split_one = split_features_target(self.df, test_size=0.25, random_state=42)
        split_two = split_features_target(self.df, test_size=0.25, random_state=42)

        X_train_one, X_test_one, y_train_one, y_test_one = split_one
        X_train_two, X_test_two, y_train_two, y_test_two = split_two

        self.assertListEqual(list(X_test_one.index), list(X_test_two.index))
        self.assertListEqual(list(y_train_one.index), list(y_train_two.index))
        self.assertEqual(len(X_train_one), 15)
        self.assertEqual(len(X_test_one), 5)
        self.assertEqual(set(y_test_one.unique()), {0, 1})
        self.assertListEqual(list(X_train_one.columns), FEATURE_COLUMNS)

    def test_preprocessor_handles_missing_and_unknown_categories(self):
        preprocessor = build_preprocessor()
        X = self.df[FEATURE_COLUMNS].copy()
        transformed = preprocessor.fit_transform(X)

        new_row = X.iloc[[0]].copy()
        new_row.loc[new_row.index[0], "gender"] = "Other"
        new_row.loc[new_row.index[0], "smoking_history"] = "No Info"
        new_row.loc[new_row.index[0], "insulin"] = np.nan
        transformed_new = preprocessor.transform(new_row)

        self.assertEqual(transformed.shape[0], len(X))
        self.assertEqual(transformed_new.shape[0], 1)
        self.assertEqual(transformed_new.shape[1], transformed.shape[1])

    def test_scale_pos_weight_uses_training_class_ratio(self):
        y = pd.Series([0, 0, 0, 1])
        self.assertEqual(compute_scale_pos_weight(y), 3.0)
        self.assertEqual(compute_scale_pos_weight(pd.Series([0, 0])), 1.0)

    def test_evaluation_metrics_are_computed_correctly(self):
        metrics = evaluate_binary_classifier(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            y_score=[0.1, 0.7, 0.8, 0.9],
        )

        self.assertAlmostEqual(metrics.accuracy, 0.75)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.specificity, 0.5)
        self.assertIsNotNone(metrics.roc_auc)
        self.assertIsNotNone(metrics.pr_auc)


if __name__ == "__main__":
    unittest.main()
