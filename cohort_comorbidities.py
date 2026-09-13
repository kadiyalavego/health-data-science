import sqlite3
import pandas as pd

conn = sqlite3.connect("clinical_data.db")

sql_query = """
WITH Comorbidities AS (
    -- Aggregate secondary diagnosis flags per admission
    SELECT 
        hadm_id,
        MAX(CASE WHEN icd10_code LIKE 'E11%' THEN 1 ELSE 0 END) AS flag_diabetes,
        MAX(CASE WHEN icd10_code LIKE 'I10%' THEN 1 ELSE 0 END) AS flag_hypertension,
        MAX(CASE WHEN icd10_code LIKE 'J44%' THEN 1 ELSE 0 END) AS flag_copd
    FROM diagnoses
    WHERE priority > 1
    GROUP BY hadm_id
)
SELECT 
    p.patient_id,
    p.gender,
    CAST((strftime('%Y', a.admit_time) - strftime('%Y', p.dob)) AS INT) AS age,
    ROUND((julianday(a.discharge_time) - julianday(a.admit_time)), 2) AS los_days,
    
    -- Baseline Vitals
    MAX(CASE WHEN v.vital_name = 'HeartRate' THEN v.valuenum END) AS hr_baseline,
    MAX(CASE WHEN v.vital_name = 'SysBP' THEN v.valuenum END) AS sbp_baseline,
    
    -- Comorbidity Binary Flags (Default to 0 if no record exists)
    COALESCE(c.flag_diabetes, 0) AS has_diabetes,
    COALESCE(c.flag_hypertension, 0) AS has_hypertension,
    COALESCE(c.flag_copd, 0) AS has_copd

FROM admissions a
INNER JOIN patients p 
    ON a.patient_id = p.patient_id
INNER JOIN diagnoses d 
    ON a.hadm_id = d.hadm_id
LEFT JOIN vitals v 
    ON a.hadm_id = v.hadm_id
LEFT JOIN Comorbidities c 
    ON a.hadm_id = c.hadm_id
WHERE d.priority = 1 
  AND d.icd10_code LIKE 'I50%'
GROUP BY 
    p.patient_id, p.gender, age, los_days, 
    c.flag_diabetes, c.flag_hypertension, c.flag_copd
ORDER BY a.hadm_id;
"""

df_final = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- FULL CLINICAL MODELING COHORT (WEEK 01 COMPLETE) ---")
print(df_final.to_string(index=False))
