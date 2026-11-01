# Tumor Classification Lab

## Problem statement

Train and compare five classification models that identify whether a breast
tumor is malignant or benign. The Streamlit application evaluates the saved
models on labeled CSV test data, presents a metric comparison, and displays a
confusion matrix and classification report. Test data is never used to retrain
the models.

## Dataset description

This project uses the Wisconsin Diagnostic Breast Cancer dataset (WDBC), a
public binary classification dataset originally donated to the UCI Machine
Learning Repository. It contains 569 observations and 30 numeric features
computed from digitized fine needle aspirate images. The target is `diagnosis`,
with `malignant` as the positive class and `benign` as the negative class.

Dataset source: [UCI Machine Learning Repository — Breast Cancer Wisconsin
(Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
The dataset is loaded through scikit-learn's `load_breast_cancer` dataset
loader. `dataset.csv` contains the complete 569-row dataset, satisfying the
assignment's minimum instance size. A stratified 80/20 split with random state
42 is used: 455 rows for training and 114 rows for held-out testing.
`test_data.csv` contains only those 114 test rows. The app evaluates this
holdout by default; keeping the training rows out of `test_data.csv` avoids
testing the models on data they were trained on.

## GitHub Repository Link

**Repository:** `https://github.com/2026aj05612/2026AJ05612_Assignment2`

## Models used

Models are trained on the same training split. Feature standardization is
included in the Logistic Regression, K-Nearest Neighbors, and Support Vector
Machine pipelines. The Decision Tree and Gaussian Naive Bayes models use the
original numeric features. Precision, recall, and F1 use malignant as the
positive class; AUC uses the malignant-class probability.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong results across the metrics and the highest AUC (0.9960), indicating excellent ranking of malignant cases. |
| Decision Tree | Lowest accuracy and AUC in this comparison. Its results are easier to explain, but weaker than the other classifiers on this holdout set. |
| K-Nearest Neighbors | Performs well overall, though malignant recall is lower than the linear and SVM models. |
| Gaussian Naive Bayes | Has perfect malignant precision on this split, but the lowest malignant recall among the five models, so it misses more malignant cases. |
| Support Vector Machine | Best overall holdout result: highest accuracy, recall, F1, and MCC, while retaining high precision and AUC. |

**Overall winner:** Support Vector Machine on this fixed holdout split. Logistic
Regression has the highest AUC. These scores are from a single split and should
not be interpreted as a guarantee of performance on new populations.

## Streamlit app

The app loads the five pre-trained pipelines from `model/`, uses bundled
`test_data.csv` by default, and also accepts an uploaded CSV. The input file
must include all 30 feature columns and a `diagnosis` column containing only
`malignant` and `benign`. It displays all six metrics for all five models,
along with a confusion matrix and classification report for the selected
model. No model is retrained in the app.

**Live Streamlit app:** `https://2026aj05612assignment2-l5ai6yugnd29vdsrcqbihw.streamlit.app/`

## Repository contents

```text
Assignment2_repo/
|-- app.py
|-- modeling.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- dataset.csv
|-- test_data.csv
|-- model/
|   |-- decision_tree.joblib
|   |-- evaluation_metrics.json
|   |-- gaussian_nb.joblib
|   |-- knn.joblib
|   |-- logistic_regression.joblib
|   `-- svm.joblib
`-- tests/
    `-- test_modeling.py
```

## Run locally

Python 3.11 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

To run the model tests:

```powershell
python -m unittest discover -s tests -v
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), create a new app
   from that repository and select `app.py`.
3. Wait for the app to finish deploying, open its public URL, and verify that
   both the bundled holdout and an uploaded labeled CSV evaluate correctly.
4. 