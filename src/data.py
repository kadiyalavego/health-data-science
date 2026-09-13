import sqlite3
import pandas as pd

def load_multimodal_cohort(db_path: str = "clinical_data.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    sql_query = """
    WITH Comorbidities AS (
        SELECT hadm_id,
            MAX(CASE WHEN icd10_code LIKE 'E11%' THEN 1 ELSE 0 END) AS flag_diabetes,
            MAX(CASE WHEN icd10_code LIKE 'I10%' THEN 1 ELSE 0 END) AS flag_hypertension,
            MAX(CASE WHEN icd10_code LIKE 'J44%' THEN 1 ELSE 0 END) AS flag_copd
        FROM diagnoses WHERE priority > 1 GROUP BY hadm_id
    ),
    RankedLabs AS (
        SELECT 
            l.hadm_id, l.lab_name, l.valuenum, l.chart_time,
            ROW_NUMBER() OVER (PARTITION BY l.hadm_id, l.lab_name ORDER BY l.chart_time ASC) AS rn_first,
            ROW_NUMBER() OVER (PARTITION BY l.hadm_id, l.lab_name ORDER BY l.chart_time DESC) AS rn_last
        FROM labs l
        INNER JOIN admissions a ON l.hadm_id = a.hadm_id
        WHERE julianday(l.chart_time) - julianday(a.admit_time) <= 2.5
    ),
    AggregatedLabs AS (
        SELECT 
            hadm_id,
            MAX(CASE WHEN lab_name = 'Creatinine' AND rn_first = 1 THEN valuenum END) AS cr_baseline,
            MAX(CASE WHEN lab_name = 'Creatinine' AND rn_last = 1 THEN valuenum END) AS cr_48h,
            MAX(CASE WHEN lab_name = 'NT-proBNP' AND rn_first = 1 THEN valuenum END) AS bnp_baseline,
            MAX(CASE WHEN lab_name = 'NT-proBNP' AND rn_last = 1 THEN valuenum END) AS bnp_48h
        FROM RankedLabs GROUP BY hadm_id
    )
    SELECT 
        a.hadm_id,
        CAST((strftime('%Y', a.admit_time) - strftime('%Y', p.dob)) AS INT) AS age,
        MAX(CASE WHEN v.vital_name = 'HeartRate' THEN v.valuenum END) AS hr_baseline,
        MAX(CASE WHEN v.vital_name = 'SysBP' THEN v.valuenum END) AS sbp_baseline,
        COALESCE(c.flag_diabetes, 0) AS has_diabetes,
        COALESCE(c.flag_hypertension, 0) AS has_hypertension,
        COALESCE(c.flag_copd, 0) AS has_copd,
        l.cr_baseline,
        l.cr_48h,
        l.bnp_baseline,
        l.bnp_48h,
        ROUND((julianday(a.discharge_time) - julianday(a.admit_time)), 2) AS los_days
    FROM admissions a
    INNER JOIN patients p ON a.patient_id = p.patient_id
    INNER JOIN diagnoses d ON a.hadm_id = d.hadm_id
    LEFT JOIN vitals v ON a.hadm_id = v.hadm_id
    LEFT JOIN Comorbidities c ON a.hadm_id = c.hadm_id
    LEFT JOIN AggregatedLabs l ON a.hadm_id = l.hadm_id
    WHERE d.priority = 1 AND d.icd10_code LIKE 'I50%'
    GROUP BY a.hadm_id, age, los_days, c.flag_diabetes, c.flag_hypertension, c.flag_copd,
             l.cr_baseline, l.cr_48h, l.bnp_baseline, l.bnp_48h;
    """
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df