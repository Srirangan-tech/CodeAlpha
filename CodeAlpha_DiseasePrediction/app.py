from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ART=Path("artifacts")
st.set_page_config(page_title="Disease Prediction",page_icon="🩺")
st.title("🩺 Disease Prediction from Medical Data")
st.caption("CodeAlpha Machine Learning Task 4 — Breast Cancer Wisconsin dataset")
st.error("Educational model only. It is not a medical device and cannot diagnose cancer.")

required=[ART/"best_disease_model.joblib",ART/"feature_schema.json",ART/"meta.json"]
if not all(p.exists() for p in required): st.info("Train the model first with `python train.py`."); st.stop()
model=joblib.load(required[0]); schema=json.loads(required[1].read_text()); meta=json.loads(required[2].read_text()); features=meta["features"]
values={}
with st.expander("Patient / sample measurements",expanded=True):
    for f in features:
        s=schema[f]; values[f]=st.number_input(f.title(),min_value=float(s["min"]),max_value=float(s["max"]),value=float(s["default"]),format="%.5f")
if st.button("Run prediction",type="primary"):
    X=pd.DataFrame([[values[f] for f in features]],columns=features); prob_benign=float(model.predict_proba(X)[0,1]); pred=int(prob_benign>=.5)
    label="Benign" if pred==1 else "Malignant"
    st.subheader(f"Model output: {label}")
    st.metric("Probability of benign class",f"{prob_benign:.1%}")
    st.caption("This result is for machine-learning demonstration only and must not be interpreted as a diagnosis.")
