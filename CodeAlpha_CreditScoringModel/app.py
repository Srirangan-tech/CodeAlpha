from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")
FEATURES = [
    "annual_income", "monthly_debt", "credit_utilization",
    "missed_payments_12m", "credit_history_years", "employment_years",
    "existing_loans", "savings_balance", "age"
]


def pct(value):
    return f"{value:.1%}"


def money(value):
    return f"${value:,.0f}"


st.set_page_config(
    page_title="CreditLens • Credit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
      .stApp {
        background:
          radial-gradient(circle at 10% 5%, rgba(99,102,241,.12), transparent 28%),
          radial-gradient(circle at 92% 8%, rgba(14,165,233,.10), transparent 26%);
      }
      .hero {
        padding: 1.55rem 1.7rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 18px 50px rgba(2,6,23,.18);
        margin-bottom: 1.3rem;
      }
      .eyebrow {
        color: #a5b4fc; font-size: .78rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase;
      }
      .hero h1 {color: white; margin: .25rem 0 .35rem 0; font-size: 2.35rem;}
      .hero p {color: #cbd5e1; margin: 0; max-width: 800px; font-size: 1rem;}
      .section-label {
        color: #64748b; font-size: .76rem; font-weight: 800;
        letter-spacing: .11em; text-transform: uppercase; margin-bottom: .25rem;
      }
      div[data-testid="stMetric"] {
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 18px;
        padding: .9rem 1rem;
        background: rgba(255,255,255,.025);
      }
      div[data-testid="stForm"] {
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 22px;
        padding: 1.05rem 1.15rem 1.25rem 1.15rem;
        background: rgba(255,255,255,.02);
      }
      .result-good, .result-mid, .result-risk {
        padding: 1.2rem 1.35rem; border-radius: 22px;
        border: 1px solid rgba(148,163,184,.18); margin-bottom: 1rem;
      }
      .result-good {background: linear-gradient(135deg, rgba(16,185,129,.16), rgba(5,150,105,.06));}
      .result-mid {background: linear-gradient(135deg, rgba(245,158,11,.16), rgba(217,119,6,.06));}
      .result-risk {background: linear-gradient(135deg, rgba(239,68,68,.16), rgba(220,38,38,.06));}
      .result-title {font-size: 1.55rem; font-weight: 800; margin-bottom: .25rem;}
      .muted {color: #64748b; font-size: .92rem;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CodeAlpha • Machine Learning Task 1</div>
      <h1>💳 CreditLens</h1>
      <p>
        An interactive machine-learning creditworthiness demo. Enter a financial profile,
        run the assessment, and inspect the model probability, profile ratios and evaluation results.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_path = ARTIFACTS / "best_credit_model.joblib"
schema_path = ARTIFACTS / "feature_schema.json"
metrics_path = ARTIFACTS / "metrics.json"

if not model_path.exists() or not schema_path.exists():
    st.error("Model artifacts were not found.")
    st.code("python train.py", language="bash")
    st.caption("Run the training command from this project folder, then refresh the page.")
    st.stop()

model = joblib.load(model_path)
schema = json.loads(schema_path.read_text())
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
best_metrics = max(metrics, key=lambda row: row["roc_auc"]) if metrics else None

with st.sidebar:
    st.markdown("### 💳 CreditLens")
    st.caption("Interactive ML demo")
    st.divider()
    st.success("Model loaded successfully")
    if best_metrics:
        st.metric("Selected model", best_metrics["model"])
        st.metric("Test ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
    st.divider()
    st.markdown("**How to use**")
    st.caption(
        "1. Enter the applicant profile.\n\n"
        "2. Click **Run credit assessment**.\n\n"
        "3. Review the model probability and evaluation tabs."
    )
    st.divider()
    st.warning(
        "Educational demonstration only. This synthetic-data model must not be used "
        "for real lending decisions."
    )

left, right = st.columns([1.7, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Applicant profile</div>', unsafe_allow_html=True)
    st.subheader("Enter financial details")

    values = {}
    with st.form("credit_form"):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("##### Income & obligations")
            values["annual_income"] = st.number_input(
                "Annual income",
                min_value=float(schema["annual_income"]["min"]),
                max_value=float(schema["annual_income"]["max"]),
                value=float(schema["annual_income"]["default"]),
                step=1000.0,
                format="%.0f",
                help="Total yearly income.",
            )
            values["monthly_debt"] = st.number_input(
                "Monthly debt payments",
                min_value=float(schema["monthly_debt"]["min"]),
                max_value=float(schema["monthly_debt"]["max"]),
                value=float(schema["monthly_debt"]["default"]),
                step=100.0,
                format="%.0f",
            )
            values["savings_balance"] = st.number_input(
                "Savings balance",
                min_value=float(schema["savings_balance"]["min"]),
                max_value=float(schema["savings_balance"]["max"]),
                value=float(schema["savings_balance"]["default"]),
                step=500.0,
                format="%.0f",
            )
            values["existing_loans"] = st.number_input(
                "Existing loans",
                min_value=int(schema["existing_loans"]["min"]),
                max_value=int(schema["existing_loans"]["max"]),
                value=int(schema["existing_loans"]["default"]),
                step=1,
            )
            values["age"] = st.number_input(
                "Age",
                min_value=int(schema["age"]["min"]),
                max_value=int(schema["age"]["max"]),
                value=int(schema["age"]["default"]),
                step=1,
            )

        with c2:
            st.markdown("##### Credit & stability")
            values["credit_utilization"] = st.slider(
                "Credit utilization",
                min_value=float(schema["credit_utilization"]["min"]),
                max_value=float(schema["credit_utilization"]["max"]),
                value=float(schema["credit_utilization"]["default"]),
                step=0.01,
                format="%.2f",
                help="0.30 means 30% of available revolving credit is used.",
            )
            values["missed_payments_12m"] = st.number_input(
                "Missed payments (last 12 months)",
                min_value=int(schema["missed_payments_12m"]["min"]),
                max_value=int(schema["missed_payments_12m"]["max"]),
                value=int(schema["missed_payments_12m"]["default"]),
                step=1,
            )
            values["credit_history_years"] = st.number_input(
                "Credit history (years)",
                min_value=float(schema["credit_history_years"]["min"]),
                max_value=float(schema["credit_history_years"]["max"]),
                value=float(schema["credit_history_years"]["default"]),
                step=0.5,
                format="%.1f",
            )
            values["employment_years"] = st.number_input(
                "Employment history (years)",
                min_value=float(schema["employment_years"]["min"]),
                max_value=float(schema["employment_years"]["max"]),
                value=float(schema["employment_years"]["default"]),
                step=0.5,
                format="%.1f",
            )
            st.write("")
            st.caption(f"Current utilization: **{values['credit_utilization']:.0%}**")

        submitted = st.form_submit_button(
            "Run credit assessment",
            type="primary",
            use_container_width=True,
        )

with right:
    st.markdown('<div class="section-label">Live profile snapshot</div>', unsafe_allow_html=True)
    st.subheader("Key ratios")

    monthly_income = values["annual_income"] / 12 if values["annual_income"] else 0
    dti = values["monthly_debt"] / monthly_income if monthly_income else 0
    savings_ratio = values["savings_balance"] / values["annual_income"] if values["annual_income"] else 0

    a, b = st.columns(2)
    a.metric("Debt / monthly income", pct(dti))
    b.metric("Credit utilization", pct(values["credit_utilization"]))

    c, d = st.columns(2)
    c.metric("Savings / annual income", pct(savings_ratio))
    d.metric("Missed payments", int(values["missed_payments_12m"]))

    st.caption(
        "These are profile summaries for interpretation. The prediction itself uses the full feature set."
    )
    st.divider()

    snapshot = pd.DataFrame(
        {
            "Indicator": ["Income", "Savings", "Credit history", "Employment history"],
            "Value": [
                money(values["annual_income"]),
                money(values["savings_balance"]),
                f"{values['credit_history_years']:.1f} years",
                f"{values['employment_years']:.1f} years",
            ],
        }
    )
    st.dataframe(snapshot, use_container_width=True, hide_index=True)

if submitted:
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    probability = float(model.predict_proba(X)[0, 1])
    risk_probability = 1 - probability
    pred = int(probability >= 0.5)

    if probability >= 0.70:
        css_class = "result-good"
        title = "✅ Stronger model profile"
        description = "The model assigns a relatively high probability of creditworthiness to this profile."
    elif probability >= 0.50:
        css_class = "result-mid"
        title = "🟡 Moderate model profile"
        description = "The model leans toward creditworthy, but the result is closer to the decision threshold."
    else:
        css_class = "result-risk"
        title = "⚠️ Higher-risk model profile"
        description = "The model assigns a lower probability of creditworthiness to this profile."

    st.divider()
    st.markdown('<div class="section-label">Assessment result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="result-title">{title}</div>
          <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Creditworthiness probability", pct(probability))
    r2.metric("Risk probability", pct(risk_probability))
    r3.metric("Model decision", "Creditworthy" if pred else "Higher risk")
    st.progress(max(0.0, min(1.0, probability)), text=f"Model probability: {probability:.1%}")

    score_tab, performance_tab, about_tab = st.tabs(
        ["Score details", "Model performance", "About this demo"]
    )

    with score_tab:
        summary = pd.DataFrame(
            {
                "Measure": [
                    "Annual income", "Monthly debt", "Debt-to-income", "Credit utilization",
                    "Missed payments", "Credit history", "Employment history", "Existing loans", "Savings"
                ],
                "Value": [
                    money(values["annual_income"]), money(values["monthly_debt"]), pct(dti),
                    pct(values["credit_utilization"]), str(int(values["missed_payments_12m"])),
                    f"{values['credit_history_years']:.1f} years",
                    f"{values['employment_years']:.1f} years",
                    str(int(values["existing_loans"])), money(values["savings_balance"])
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.info(
            "The 50% threshold is used only for this educational classifier. "
            "The probability shown here is not a real-world credit score."
        )

    with performance_tab:
        if metrics:
            metrics_df = pd.DataFrame(metrics).rename(
                columns={
                    "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
                    "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"
                }
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("##### ROC curve")
            roc = ARTIFACTS / "roc_curve.png"
            if roc.exists():
                st.image(str(roc), use_container_width=True)
        with chart2:
            st.markdown("##### Confusion matrix")
            cm = ARTIFACTS / "confusion_matrix.png"
            if cm.exists():
                st.image(str(cm), use_container_width=True)

    with about_tab:
        st.markdown(
            """
            **This project demonstrates:**
            - Financial-history feature engineering
            - Logistic Regression, Decision Tree and Random Forest
            - Automatic best-model selection using ROC-AUC
            - Accuracy, Precision, Recall, F1 and ROC-AUC evaluation
            - Interactive Streamlit inference

            The training dataset is synthetic and reproducible, so this application is intended
            as an ML workflow demonstration rather than a lending product.
            """
        )
else:
    st.info("Adjust the applicant profile and select **Run credit assessment** to see the result.")
