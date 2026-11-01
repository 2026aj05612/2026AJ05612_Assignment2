"""Test model construction and required evaluation metric calculations."""

import unittest

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from modeling import MODEL_NAMES, build_models, calculate_metrics


class ModelEvaluationTests(unittest.TestCase):
    def test_all_five_models_produce_required_metrics(self) -> None:
        dataset = load_breast_cancer(as_frame=True)
        features = dataset.data
        labels = pd.Series(
            dataset.target.map(dict(enumerate(dataset.target_names)))
        )
        # Match the training script's split so the test checks the same evaluation setup.
        train_x, test_x, train_y, test_y = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        expected = {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}

        models = build_models()
        self.assertEqual(len(models), 5)
        self.assertEqual(set(models), set(MODEL_NAMES))
        for model in models.values():
            with self.subTest(model=type(model).__name__):
                model.fit(train_x, train_y)
                metrics = calculate_metrics(model, test_x, test_y)
                self.assertEqual(set(metrics), expected)
                self.assertTrue(all(0 <= metrics[key] <= 1 for key in expected - {"MCC"}))
                self.assertTrue(-1 <= metrics["MCC"] <= 1)


if __name__ == "__main__":
    unittest.main()
