import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve
)

# 1. Query clinical cohort
conn = sqlite3.connect("clinical_data.db")
sql_query = """
WITH Comorbidities AS (
    SELECT hadm_id,
        MAX(CASE WHEN icd10_code LIKE 'E11%' THEN 1 ELSE 0 END) AS flag_diabetes,
        MAX(CASE WHEN icd10_code LIKE 'I10%' THEN 1 ELSE 0 END) AS flag_hypertension,
        MAX(CASE WHEN icd10_code LIKE 'J44%' THEN 1 ELSE 0 END) AS flag_copd
    FROM diagnoses WHERE priority > 1 GROUP BY hadm_id
)
SELECT 
    CAST((strftime('%Y', a.admit_time) - strftime('%Y', p.dob)) AS INT) AS age,
    MAX(CASE WHEN v.vital_name = 'HeartRate' THEN v.valuenum END) AS hr_baseline,
    MAX(CASE WHEN v.vital_name = 'SysBP' THEN v.valuenum END) AS sbp_baseline,
    COALESCE(c.flag_diabetes, 0) AS has_diabetes,
    COALESCE(c.flag_hypertension, 0) AS has_hypertension,
    COALESCE(c.flag_copd, 0) AS has_copd,
    ROUND((julianday(a.discharge_time) - julianday(a.admit_time)), 2) AS los_days
FROM admissions a
INNER JOIN patients p ON a.patient_id = p.patient_id
INNER JOIN diagnoses d ON a.hadm_id = d.hadm_id
LEFT JOIN vitals v ON a.hadm_id = v.hadm_id
LEFT JOIN Comorbidities c ON a.hadm_id = c.hadm_id
WHERE d.priority = 1 AND d.icd10_code LIKE 'I50%'
GROUP BY p.patient_id, age, los_days, c.flag_diabetes, c.flag_hypertension, c.flag_copd;
"""
df = pd.read_sql_query(sql_query, conn)
conn.close()

# 2. Setup binary outcome (LOS > 7 days)
df['prolonged_los'] = (df['los_days'] > 7.0).astype(int)

features = ['age', 'hr_baseline', 'sbp_baseline', 'has_diabetes', 'has_hypertension', 'has_copd']
X = df[features]
y = df['prolonged_los']

# 3. Stratified 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Fit scaler on train data only (preventing data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 6. Generate Predictions & Probabilities on Holdout Test Set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 7. Print Clinical Diagnostics
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("--- HOLDOUT TEST SET PERFORMANCE (N = 200) ---")
print(f"ROC-AUC Score: {auc:.3f}\n")
print("Confusion Matrix:")
print(f"  [TN: {cm[0,0]:3d}  |  FP: {cm[0,1]:3d}]")
print(f"  [FN: {cm[1,0]:3d}  |  TP: {cm[1,1]:3d}]\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal LOS (<=7d)', 'Prolonged LOS (>7d)']))

# 8. Clinical Odds Ratios
or_df = pd.DataFrame({
    'Feature': features,
    'Log-Odds (Beta)': model.coef_[0],
    'Odds Ratio (OR)': np.exp(model.coef_[0])
}).sort_values(by='Odds Ratio (OR)', ascending=False)

print("Clinical Odds Ratios (Multiplicative risk per SD / category):")
print(or_df.to_string(index=False))