import pandas as pd
from typing import Tuple, List

FEATURE_COLS = [
    'age', 'hr_baseline', 'sbp_baseline',
    'has_diabetes', 'has_hypertension', 'has_copd',
    'cr_baseline', 'cr_delta', 'cr_measured_48h',
    'bnp_baseline'
]

def engineer_clinical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    data = df.copy()
    data['target_prolonged_los'] = (data['los_days'] > 7.0).astype(int)
    data['cr_measured_48h'] = (~data['cr_48h'].isna()).astype(int)
    data['cr_48h_clean'] = data['cr_48h'].fillna(data['cr_baseline'])
    data['cr_delta'] = data['cr_48h_clean'] - data['cr_baseline']
    data['bnp_baseline'] = data['bnp_baseline'].fillna(data['bnp_baseline'].median())
    
    X = data[FEATURE_COLS]
    y = data['target_prolonged_los']
    return X, y, FEATURE_COLS
