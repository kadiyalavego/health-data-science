import sqlite3
import pandas as pd

conn = sqlite3.connect("clinical_data.db")

sql_query = """
SELECT 
    p.patient_id,
    p.gender,
    -- Calculate age at admission in years
    CAST((strftime('%Y', a.admit_time) - strftime('%Y', p.dob)) AS INT) AS age_at_admission,
    a.hadm_id,
    a.admission_type,
    -- Calculate length of stay (LOS) in days
    ROUND((julianday(a.discharge_time) - julianday(a.admit_time)), 2) AS los_days,
    d.icd10_code,
    d.icd10_title
FROM admissions a
INNER JOIN patients p 
    ON a.patient_id = p.patient_id
INNER JOIN diagnoses d 
    ON a.hadm_id = d.hadm_id
WHERE d.priority = 1                      -- Primary diagnosis only
  AND d.icd10_code LIKE 'I50%'            -- Heart Failure umbrella code
ORDER BY a.hadm_id;
"""

df_cohort = pd.read_sql_query(sql_query, conn)
conn.close()

print("--- EXTRACTED CLINICAL COHORT ---")
print(df_cohort.to_string(index=False))

print("\n--- COHORT SUMMARY METRICS ---")
print(f"Total Cohort Size:       {len(df_cohort)}")
print(f"Mean Age:                {df_cohort['age_at_admission'].mean():.1f} years")
print(f"Mean Length of Stay:     {df_cohort['los_days'].mean():.2f} days")