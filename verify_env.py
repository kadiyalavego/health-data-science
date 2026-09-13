import sqlite3
import pandas as pd
import numpy as np
import sklearn
import lifelines

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
cursor.execute("CREATE TABLE test_patients (patient_id INT, age INT);")
cursor.execute("INSERT INTO test_patients VALUES (1, 58), (2, 64);")
conn.commit()

df = pd.read_sql_query("SELECT * FROM test_patients", conn)
conn.close()

print("--- ENVIRONMENT VERIFICATION SUCCESSFUL ---")
print(f"Pandas version:     {pd.__version__}")
print(f"Scikit-Learn:       {sklearn.__version__}")
print(f"Lifelines version:  {lifelines.__version__}")
print("\nSample clinical cohort query result:")
print(df)