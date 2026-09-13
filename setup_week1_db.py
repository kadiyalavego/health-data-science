import sqlite3
import pandas as pd
from datetime import datetime

# Connect to (or create) local database file
conn = sqlite3.connect("clinical_data.db")
cursor = conn.cursor()

# 1. Create Relational Tables
cursor.executescript("""
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS admissions;
DROP TABLE IF EXISTS diagnoses;
DROP TABLE IF EXISTS vitals;

CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    gender TEXT,
    dob DATE
);

CREATE TABLE admissions (
    hadm_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    admit_time TIMESTAMP,
    discharge_time TIMESTAMP,
    admission_type TEXT,
    discharge_location TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE diagnoses (
    diag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hadm_id INTEGER,
    icd10_code TEXT,
    icd10_title TEXT,
    priority INTEGER, -- 1 = Primary diagnosis, 2+ = Secondary
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);

CREATE TABLE vitals (
    vital_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hadm_id INTEGER,
    chart_time TIMESTAMP,
    vital_name TEXT,
    valuenum REAL,
    FOREIGN KEY(hadm_id) REFERENCES admissions(hadm_id)
);
""")

# 2. Populate with Representative Synthetic Inpatient Data
cursor.executescript("""
INSERT INTO patients VALUES 
(101, 'M', '1952-04-12'),
(102, 'F', '1968-11-03'),
(103, 'F', '1945-01-20'),
(104, 'M', '1980-07-15'),
(105, 'M', '1961-09-30');

INSERT INTO admissions VALUES
(5001, 101, '2026-01-10 08:30:00', '2026-01-15 14:00:00', 'EMERGENCY', 'HOME'),
(5002, 102, '2026-01-12 11:15:00', '2026-01-18 10:00:00', 'ELECTIVE', 'HOME'),
(5003, 103, '2026-02-01 19:45:00', '2026-02-14 16:30:00', 'EMERGENCY', 'SNF'),
(5004, 104, '2026-02-05 06:00:00', '2026-02-06 18:00:00', 'URGENT', 'HOME'),
(5005, 105, '2026-02-10 14:20:00', '2026-02-22 11:00:00', 'EMERGENCY', 'HOME');

INSERT INTO diagnoses (hadm_id, icd10_code, icd10_title, priority) VALUES
(5001, 'I50.9', 'Heart failure, unspecified', 1),
(5001, 'E11.9', 'Type 2 diabetes mellitus', 2),
(5002, 'M16.1', 'Primary osteoarthritis, right hip', 1),
(5003, 'I50.22', 'Chronic systolic heart failure', 1),
(5003, 'I10', 'Essential (primary) hypertension', 2),
(5003, 'N18.3', 'Chronic kidney disease, stage 3', 3),
(5004, 'S83.5', 'Sprain of cruciate ligament of knee', 1),
(5005, 'I50.9', 'Heart failure, unspecified', 1),
(5005, 'J44.1', 'COPD with acute exacerbation', 2);

INSERT INTO vitals (hadm_id, chart_time, vital_name, valuenum) VALUES
(5001, '2026-01-10 09:00:00', 'HeartRate', 104),
(5001, '2026-01-10 09:00:00', 'SysBP', 145),
(5003, '2026-02-01 20:00:00', 'HeartRate', 92),
(5003, '2026-02-01 20:00:00', 'SysBP', 160),
(5005, '2026-02-10 15:00:00', 'HeartRate', 118),
(5005, '2026-02-10 15:00:00', 'SysBP', 138);
""")

conn.commit()
conn.close()
print("Database 'clinical_data.db' created successfully with tables: patients, admissions, diagnoses, vitals.")