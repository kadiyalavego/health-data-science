import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

df['prolonged_los'] = (df['los_days'] > 7.0).astype(int)

features = ['age', 'hr_baseline', 'sbp_baseline', 'has_diabetes', 'has_hypertension', 'has_copd']
X = df[features]
y = df['prolonged_los']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)
accuracy = model.score(X_scaled, y)

print("--- MODEL TRAINING COMPLETE ---")
print(f"Model Accuracy (Training): {accuracy * 100:.1f}%\n")
print("Feature Coefficients (Log-Odds Impact on Prolonged LOS):")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature.ljust(18)}: {coef:+.3f}")