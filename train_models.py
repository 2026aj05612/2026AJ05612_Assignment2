from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from modeling import (
    MODEL_NAMES,
    TARGET_COLUMN,
    build_models,
    calculate_metrics,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
DATASET_PATH = ROOT / "dataset.csv"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    labels = pd.Series(
        dataset.target.map(dict(enumerate(dataset.target_names))),
        name=TARGET_COLUMN,
    )
    full_dataset = features.copy()
    full_dataset[TARGET_COLUMN] = labels
    full_dataset.to_csv(DATASET_PATH, index=False)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    MODEL_DIR.mkdir(exist_ok=True)

    test_data = test_features.copy()
    test_data[TARGET_COLUMN] = test_labels
    test_data.to_csv(TEST_DATA_PATH, index=False)

    results: dict[str, dict[str, float]] = {}
    for key, model in build_models().items():
        model.fit(train_features, train_labels)
        joblib.dump(model, MODEL_DIR / f"{key}.joblib")
        results[MODEL_NAMES[key]] = calculate_metrics(
            model, test_features, test_labels
        )

    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} models to {MODEL_DIR}")
    print(f"Saved full dataset with {len(full_dataset)} rows to {DATASET_PATH}")
    print(f"Saved {len(test_data)} holdout rows to {TEST_DATA_PATH}")
    print(pd.DataFrame.from_dict(results, orient="index").to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
