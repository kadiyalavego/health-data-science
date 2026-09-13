import sqlite3
import pandas as pd

conn = sqlite3.connect("clinical_data.db")

sql_query = """
SELECT 
    p.patient_id,
    p.gender,
    CAST((strftime('%Y', a.admit_time) - strftime('%Y', p.dob)) AS INT) AS age_at_admission,
    a.hadm_id,
    ROUND((julianday(a.discharge_time) - julianday(a.admit_time)), 2) AS los_days,
    d.icd10_code,
    
    -- Conditional Aggregation: Map vertical time-series vitals to horizontal columns
    MAX(CASE WHEN v.vital_name = 'HeartRate' THEN v.valuenum END) AS baseline_heart_rate,
    MAX(CASE WHEN v.vital_name = 'SysBP' THEN v.valuenum END) AS baseline_sys_bp

FROM admissions a
INNER JOIN patients p 
    ON a.patient_id = p.patient_id
INNER JOIN diagnoses d 
    ON a.hadm_id = d.hadm_id
LEFT JOIN vitals v 
    ON a.hadm_id = v.hadm_id
WHERE d.priority = 1 
  AND d.icd10_code LIKE 'I50%'
GROUP BY 
    p.patient_id, p.gender, age_at_admission, a.hadm_id, los_days, d.icd10_code
ORDER BY a.hadm_id;
"""

df_features = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- ANALYTIC DATASET (PATIENTS + BASELINE VITALS) ---")
print(df_features.to_string(index=False))

# Check missingness
print("\nMissing values per column:")
print(df_features.isnull().sum())