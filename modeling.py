"""Define the classifiers and evaluation metrics used by the assignment."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "diagnosis"
POSITIVE_CLASS = "malignant"
NEGATIVE_CLASS = "benign"

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}


def build_models() -> dict[str, Any]:
    # Scale distance- and margin-based models; tree and Gaussian NB use raw features.
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "gaussian_nb": GaussianNB(),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=SVC(random_state=42),
                        # Calibration enables probability-based AUC evaluation for SVC.
                        ensemble=False,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(model: Any, features: Any, labels: Any) -> dict[str, float]:
    predictions = model.predict(features)
    class_names = list(model.classes_)
    # Find this class dynamically rather than assuming a fixed probability-column order.
    positive_index = class_names.index(POSITIVE_CLASS)
    positive_probabilities = model.predict_proba(features)[:, positive_index]

    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "AUC": float(roc_auc_score(labels == POSITIVE_CLASS, positive_probabilities)),
        "Precision": float(
            precision_score(
                labels, predictions, pos_label=POSITIVE_CLASS, zero_division=0
            )
        ),
        "Recall": float(
            recall_score(labels, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "F1": float(
            f1_score(labels, predictions, pos_label=POSITIVE_CLASS, zero_division=0)
        ),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }
