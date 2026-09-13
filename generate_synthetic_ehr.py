import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
n_patients = 1000

# 1. Generate Demographics
patient_ids = np.arange(1001, 1001 + n_patients)
genders = np.random.choice(['M', 'F'], size=n_patients, p=[0.52, 0.48])
ages = np.clip(np.random.normal(loc=72, scale=11, size=n_patients).astype(int), 40, 95)
base_date = datetime(2026, 1, 1)

admit_times = [base_date + timedelta(days=int(d), hours=int(h)) 
               for d, h in zip(np.random.uniform(0, 180, n_patients), np.random.uniform(0, 23, n_patients))]

# 2. Assign Comorbidities (Probabilistic based on age)
copd = np.random.binomial(1, p=np.clip(0.15 + (ages - 60) * 0.005, 0.1, 0.45))
diabetes = np.random.binomial(1, p=0.35, size=n_patients)
hypertension = np.random.binomial(1, p=0.65, size=n_patients)

# 3. Simulate Length of Stay driven by clinical features + noise
base_los = 4.0 + 0.04 * (ages - 50) + 3.2 * copd + 1.8 * diabetes + 0.8 * hypertension
hr = np.clip(np.random.normal(82, 14, n_patients) + 8 * copd, 50, 140).round(1)
sbp = np.clip(np.random.normal(132, 18, n_patients) + 12 * hypertension, 90, 200).round(1)

# Add exponential noise to simulate real-world skewed hospital LOS
los_days = np.clip(base_los + np.random.exponential(scale=2.5, size=n_patients), 1.0, 35.0).round(2)
discharge_times = [adm + timedelta(days=float(los)) for adm, los in zip(admit_times, los_days)]

# 4. Insert into SQLite
conn = sqlite3.connect("clinical_data.db")
cursor = conn.cursor()

# Clean and recreate tables
cursor.executescript("""
DELETE FROM vitals;
DELETE FROM diagnoses;
DELETE FROM admissions;
DELETE FROM patients;
""")

for i in range(n_patients):
    pid = int(patient_ids[i])
    adm_id = 10000 + i
    dob = (admit_times[i] - timedelta(days=int(ages[i] * 365.25))).strftime('%Y-%m-%d')
    
    cursor.execute("INSERT INTO patients VALUES (?, ?, ?)", (pid, genders[i], dob))
    cursor.execute("""
        INSERT INTO admissions VALUES (?, ?, ?, ?, 'EMERGENCY', 'HOME')
    """, (adm_id, pid, admit_times[i].strftime('%Y-%m-%d %H:%M:%S'), discharge_times[i].strftime('%Y-%m-%d %H:%M:%S')))
    
    # Primary Diagnosis: Heart Failure
    cursor.execute("INSERT INTO diagnoses (hadm_id, icd10_code, icd10_title, priority) VALUES (?, 'I50.9', 'Heart failure, unspecified', 1)", (adm_id,))
    
    # Secondary diagnoses
    if copd[i]:
        cursor.execute("INSERT INTO diagnoses (hadm_id, icd10_code, icd10_title, priority) VALUES (?, 'J44.1', 'COPD with acute exacerbation', 2)", (adm_id,))
    if diabetes[i]:
        cursor.execute("INSERT INTO diagnoses (hadm_id, icd10_code, icd10_title, priority) VALUES (?, 'E11.9', 'Type 2 diabetes mellitus', 2)", (adm_id,))
    if hypertension[i]:
        cursor.execute("INSERT INTO diagnoses (hadm_id, icd10_code, icd10_title, priority) VALUES (?, 'I10', 'Essential hypertension', 2)", (adm_id,))
        
    # Baseline Vitals
    cursor.execute("INSERT INTO vitals (hadm_id, chart_time, vital_name, valuenum) VALUES (?, ?, 'HeartRate', ?)", 
                   (adm_id, admit_times[i].strftime('%Y-%m-%d %H:%M:%S'), hr[i]))
    cursor.execute("INSERT INTO vitals (hadm_id, chart_time, vital_name, valuenum) VALUES (?, ?, 'SysBP', ?)", 
                   (adm_id, admit_times[i].strftime('%Y-%m-%d %H:%M:%S'), sbp[i]))

conn.commit()
conn.close()
print(f"Successfully generated and inserted {n_patients} synthetic patient records into 'clinical_data.db'.")