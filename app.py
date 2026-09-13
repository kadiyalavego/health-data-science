import streamlit as st
import numpy as np
import pandas as pd
from src.data import load_multimodal_cohort
from src.features import engineer_clinical_features
from src.models import ClinicalRiskPipeline

st.set_page_config(page_title="Heart Failure LOS Risk Stratification", layout="wide")

st.title("🏥 Inpatient Clinical Decision Support: Heart Failure LOS Risk")
st.markdown("""
This clinical tool predicts the probability of **Prolonged Length of Stay (> 7 days)** 
for patients admitted with Heart Failure ($ICD\\text{-}10: I50$), incorporating admission baseline variables 
and 48-hour renal biomarker dynamics.
""")

@st.cache_resource
def train_cached_pipeline():
    raw_df = load_multimodal_cohort()
    X, y, feature_names = engineer_clinical_features(raw_df)
    pipeline = ClinicalRiskPipeline()
    metrics = pipeline.fit_evaluate(X, y, feature_names)
    return pipeline, metrics, feature_names

pipeline, metrics, feature_names = train_cached_pipeline()

st.sidebar.header("Patient Admission Parameters")

age = st.sidebar.slider("Age (Years)", min_value=30, max_value=95, value=72, step=1)
hr = st.sidebar.slider("Baseline Heart Rate (bpm)", min_value=40, max_value=150, value=82, step=1)
sbp = st.sidebar.slider("Baseline Systolic BP (mmHg)", min_value=80, max_value=200, value=128, step=1)

st.sidebar.subheader("Comorbid Diagnoses")
has_copd = st.sidebar.checkbox("COPD (J44)", value=False)
has_diabetes = st.sidebar.checkbox("Type 2 Diabetes (E11)", value=False)
has_hypertension = st.sidebar.checkbox("Hypertension (I10)", value=True)

st.sidebar.subheader("Laboratory Trajectories")
bnp_baseline = st.sidebar.number_input("Baseline NT-proBNP (pg/mL)", min_value=50.0, max_value=5000.0, value=450.0, step=25.0)
cr_baseline = st.sidebar.number_input("Baseline Serum Creatinine (mg/dL)", min_value=0.3, max_value=8.0, value=1.05, step=0.05)

repeat_labs = st.sidebar.checkbox("48-Hour Repeat Labs Performed?", value=True)
if repeat_labs:
    cr_48h = st.sidebar.number_input("48-Hour Serum Creatinine (mg/dL)", min_value=0.3, max_value=8.0, value=1.40, step=0.05)
    cr_delta = round(cr_48h - cr_baseline, 2)
    cr_measured_48h = 1
else:
    cr_delta = 0.0
    cr_measured_48h = 0

patient_dict = {
    'age': age,
    'hr_baseline': hr,
    'sbp_baseline': sbp,
    'has_diabetes': int(has_diabetes),
    'has_hypertension': int(has_hypertension),
    'has_copd': int(has_copd),
    'cr_baseline': cr_baseline,
    'cr_delta': cr_delta,
    'cr_measured_48h': cr_measured_48h,
    'bnp_baseline': bnp_baseline
}

input_df = pd.DataFrame([patient_dict])[feature_names]
input_scaled = pipeline.scaler.transform(input_df)
pred_prob = pipeline.model.predict_proba(input_scaled)[0, 1]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Clinical Risk Estimate")
    risk_pct = pred_prob * 100
    st.metric(label="Predicted Probability of LOS > 7 Days", value=f"{risk_pct:.1f}%")
    
    if risk_pct >= 75.0:
        st.error("⚠️ **High Risk of Prolonged Admission**: Priority discharge planning, multidisciplinary review, and specialist consultation recommended.")
    elif risk_pct >= 50.0:
        st.warning("⚡ **Moderate Risk**: Standard inpatient monitoring; reassess biomarker trajectories at next ward round.")
    else:
        st.success("✅ **Low Risk**: Favourable trajectory; candidate for early step-down and ambulatory care discharge.")

    if cr_delta >= 0.3:
        st.info(f"🔎 **Clinical Note**: Creatinine rise of $\\Delta = +{cr_delta}$ mg/dL meets criteria for Acute Kidney Injury (KDIGO Stage 1).")

with col2:
    st.subheader("Model Diagnostic Context")
    st.write(f"**Cohort Holdout ROC-AUC:** `{metrics['auc']:.3f}`")
    st.write("**Top Relative Risk Modifiers (Global Odds Ratios):**")
    top_features = metrics['odds_ratios'].head(5)
    st.dataframe(top_features[['Feature', 'Odds_Ratio']], use_container_width=True)