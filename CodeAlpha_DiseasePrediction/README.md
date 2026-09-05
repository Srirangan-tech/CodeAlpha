# CodeAlpha Task 4 — Disease Prediction from Medical Data

## Objective
Predict the possibility/class of disease from structured patient measurements.

## Dataset
This implementation uses the **Breast Cancer Wisconsin Diagnostic dataset**, available directly through scikit-learn and originally derived from UCI data.

## Models
- Logistic Regression
- RBF Support Vector Machine (SVM)
- Extra Trees
- Soft-voting ensemble

## Evaluation
Accuracy, Precision, Recall, F1-score, ROC-AUC, ROC curve and confusion matrix.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

> Educational demonstration only. This model is not a medical device and must not be used for diagnosis or treatment decisions.
