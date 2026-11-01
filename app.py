"""Streamlit interface for evaluating the saved tumor classification models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from modeling import (
    MODEL_NAMES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_COLUMN,
    calculate_metrics,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Tumor Classification Lab", page_icon="🧪", layout="wide")
st.title("Tumor Classification Lab")
st.write(
    "Compare five saved classifiers on the Wisconsin Diagnostic Breast Cancer "
    "holdout data or upload another labeled CSV. Uploaded rows are only evaluated; "
    "the models are never retrained in this app."
)

st.sidebar.header("Evaluation setup")
uploaded_file = st.sidebar.file_uploader(
    "Upload labeled test data (CSV)", type=["csv"]
)
selected_name = st.sidebar.selectbox(
    "Model details", list(MODEL_NAMES.values())
)

try:
    if uploaded_file is None:
        if not TEST_DATA_PATH.is_file():
            st.error("Bundled test_data.csv is missing from this project.")
            st.stop()
        data = pd.read_csv(TEST_DATA_PATH)
        st.caption("Using the bundled, stratified 20% holdout set (random state 42).")
    else:
        data = pd.read_csv(uploaded_file)
        st.caption(f"Evaluating uploaded file: {uploaded_file.name}")
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if TARGET_COLUMN not in data.columns:
    st.error(f"The CSV must include the target column '{TARGET_COLUMN}'.")
    st.stop()

model_paths = {
    MODEL_NAMES[key]: MODEL_DIR / f"{key}.joblib" for key in MODEL_NAMES
}
if any(not path.is_file() for path in model_paths.values()):
    st.error("One or more saved models are missing. Run `python train_models.py` first.")
    st.stop()

models = {name: joblib.load(path) for name, path in model_paths.items()}
# Use the selected saved model's feature order so every classifier gets the schema it was trained on.
expected_features = list(models[selected_name].feature_names_in_)
missing_features = [name for name in expected_features if name not in data.columns]
if missing_features:
    st.error("CSV is missing required input features: " + ", ".join(missing_features))
    st.stop()

features = data.loc[:, expected_features].apply(pd.to_numeric, errors="coerce")
if features.isna().any().any():
    st.error("Input features must be numeric and cannot contain missing values.")
    st.stop()

labels = data[TARGET_COLUMN].astype(str)
valid_labels = {POSITIVE_CLASS, NEGATIVE_CLASS}
unexpected_labels = sorted(set(labels) - valid_labels)
if unexpected_labels:
    st.error(
        f"'{TARGET_COLUMN}' must contain only 'malignant' or 'benign'; "
        f"found: {', '.join(unexpected_labels)}."
    )
    st.stop()

results = {
    name: calculate_metrics(model, features, labels)
    for name, model in models.items()
}
# Present metrics consistently regardless of the order returned by the models.
comparison = pd.DataFrame.from_dict(results, orient="index")
comparison = comparison.loc[:, METRIC_COLUMNS]

st.subheader("Model comparison")
st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
st.bar_chart(comparison)

st.subheader(f"{selected_name} — detailed evaluation")
selected_model = models[selected_name]
predictions = selected_model.predict(features)
# Keep the positive/negative class ordering consistent in both detailed outputs.
report = classification_report(
    labels,
    predictions,
    labels=[POSITIVE_CLASS, NEGATIVE_CLASS],
    target_names=["Malignant", "Benign"],
    zero_division=0,
    output_dict=True,
)
confusion = confusion_matrix(
    labels, predictions, labels=[POSITIVE_CLASS, NEGATIVE_CLASS]
)
left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix** (rows: actual, columns: predicted)")
    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual malignant", "Actual benign"],
            columns=["Predicted malignant", "Predicted benign"],
        ),
        width="stretch",
    )
with right:
    st.markdown("**Classification report**")
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        width="stretch",
    )

st.caption(
    f"Rows evaluated: {len(data):,} · Positive class for precision, recall, F1 "
    f"and AUC: {POSITIVE_CLASS}."
)
