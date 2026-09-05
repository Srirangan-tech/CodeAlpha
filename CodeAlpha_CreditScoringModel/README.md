# CodeAlpha Task 1 — Credit Scoring Model

## Objective
Predict an individual's creditworthiness from past financial data.

## What this project includes
- Financial-history feature engineering
- Logistic Regression, Random Forest, Extra Trees and a weighted soft-voting ensemble
- Precision, Recall, F1-score, Accuracy and ROC-AUC
- ROC curve and confusion matrix generation
- Automatic selection and saving of the best model
- Streamlit interface for interactive predictions

## Features
`annual_income`, `monthly_debt`, `credit_utilization`, `missed_payments_12m`, `credit_history_years`, `employment_years`, `existing_loans`, `savings_balance`, `age`

The default training command creates a reproducible synthetic financial dataset so the project runs immediately. For a real internship submission, you can replace the generated dataset with a real anonymized credit dataset while keeping the same training/evaluation pipeline.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training outputs are stored under `artifacts/`.

> Educational demonstration only; do not use this model for real lending decisions.


## Revised architecture
The model comparison now includes Logistic Regression, Random Forest, Extra Trees, and a weighted soft-voting ensemble.
