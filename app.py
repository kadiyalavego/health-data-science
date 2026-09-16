import streamlit as st
import os

st.set_page_config(
    page_title="Inpatient Heart Failure Clinical CDS",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Inpatient Heart Failure Risk Stratification Pipeline")
st.markdown("### Clinical Decision Support & Real-Time Biomarker Surveillance")

st.markdown("""
This clinical decision tool utilizes multimodal Electronic Health Record (EHR) features,
including dynamic creatinine shifts (KDIGO AKI criteria), acute biomarker elevation (BNP),
and baseline demographic risk indicators.
""")

# Sidebar Inputs for Patient Triage
st.sidebar.header("Patient Admission Parameters")

age = st.sidebar.slider("Age (years)", min_value=18, max_value=100, value=68, step=1)
admission_creatinine = st.sidebar.number_input("Baseline Creatinine (mg/dL)", min_value=0.2, max_value=10.0, value=1.1, step=0.1)
peak_creatinine = st.sidebar.number_input("48h Peak Creatinine (mg/dL)", min_value=0.2, max_value=12.0, value=1.6, step=0.1)
bnp_level = st.sidebar.slider("Admission BNP (pg/mL)", min_value=10, max_value=5000, value=650, step=25)

st.sidebar.subheader("Comorbid Diagnoses")
has_diabetes = st.sidebar.checkbox("Diabetes Mellitus")
has_hypertension = st.sidebar.checkbox("Hypertension", value=True)
has_copd = st.sidebar.checkbox("COPD")

# Dynamic Clinical Feature Derivation (KDIGO AKI Rule)
creatinine_delta = peak_creatinine - admission_creatinine
kdigo_aki = (creatinine_delta >= 0.3) or (peak_creatinine >= 1.5 * admission_creatinine)

# Metric Tiles
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Creatinine Shift (48h Δ)", value=f"{creatinine_delta:+.2f} mg/dL")
with col2:
    aki_status = "Positive (Stage 1+)" if kdigo_aki else "Negative"
    st.metric(label="KDIGO AKI Marker", value=aki_status)
with col3:
    st.metric(label="Cardiac Load (BNP)", value=f"{bnp_level} pg/mL")

st.divider()

# Risk Stratification & Clinical Guidance
col_risk, col_details = st.columns([1, 2])

risk_score = 0.15
if age > 65: risk_score += 0.12
if kdigo_aki: risk_score += 0.28
if bnp_level > 800: risk_score += 0.25
elif bnp_level > 400: risk_score += 0.15
if has_diabetes: risk_score += 0.08
if has_copd: risk_score += 0.07

risk_pct = min(max(risk_score, 0.02), 0.98) * 100

with col_risk:
    st.subheader("Stratified Readmission Risk")
    st.markdown(f"## **{risk_pct:.1f}%**")
    if risk_pct >= 50:
        st.error("⚠️ High Risk Tier: Early post-discharge multidisciplinary follow-up indicated.")
    elif risk_pct >= 25:
        st.warning("⚡ Moderate Risk Tier: Review diuretic titration and renal function prior to discharge.")
    else:
        st.success("✅ Standard Risk Tier: Routine clinical trajectory.")

with col_details:
    st.subheader("Clinical Evidence & Cohort Insights")
    st.write(f"- **Patient Profile**: {age}-year-old presenting with baseline BNP of {bnp_level} pg/mL.")
    st.write(f"- **Renal Surveillance**: {'Acute dynamic shift detected consistent with KDIGO criteria.' if kdigo_aki else 'Stable renal profile over 48h baseline.'}")
    st.write(f"- **Active Comorbidities**: {', '.join(filter(None, ['Hypertension' if has_hypertension else '', 'Diabetes' if has_diabetes else '', 'COPD' if has_copd else ''])) or 'None documented'}.")

# Model Visualizations
st.divider()
st.subheader("Model Diagnostic & Survival Analytics")

viz_col1, viz_col2 = st.columns(2)
if os.path.exists("reports/calibration_curve.png"):
    with viz_col1:
        st.image("reports/calibration_curve.png", caption="Model Calibration Curve", use_container_width=True)

if os.path.exists("reports/decision_curve_analysis.png"):
    with viz_col2:
        st.image("reports/decision_curve_analysis.png", caption="Decision Curve Analysis (Net Benefit)", use_container_width=True)

viz_col3, viz_col4 = st.columns(2)
if os.path.exists("reports/kaplan_meier_survival.png"):
    with viz_col3:
        st.image("reports/kaplan_meier_survival.png", caption="Kaplan-Meier Survival Trajectory", use_container_width=True)

if os.path.exists("reports/shap_summary.png"):
    with viz_col4:
        st.image("reports/shap_summary.png", caption="Global SHAP Feature Attribution", use_container_width=True)