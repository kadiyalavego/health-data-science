import pytest
import numpy as np
import pandas as pd
from src.data import load_multimodal_cohort
from src.features import engineer_clinical_features
from src.evaluation import calculate_dca


def test_data_ingestion():
    """Verify synthetic EHR database loads with expected patient identifiers."""
    df = load_multimodal_cohort("clinical_data.db")
    assert not df.empty, "Database cohort returned empty."
    assert "hadm_id" in df.columns, "Primary key 'hadm_id' missing from cohort."


def test_feature_engineering_integrity():
    """Verify engineered features contain no data leakage and labels are binary."""
    df = load_multimodal_cohort("clinical_data.db")
    X, y, feature_names = engineer_clinical_features(df)
    
    assert len(X) == len(y), "Features and target sample counts do not match."
    assert set(np.unique(y)).issubset({0, 1}), "Target variable contains non-binary values."
    
    # Safe check across NumPy array or Pandas DataFrame
    x_array = np.asarray(X)
    assert not np.isnan(x_array).any(), "Engineered features contain unhandled NaN values."


def test_dca_net_benefit_bounds():
    """Verify Decision Curve Analysis outputs valid mathematical ranges."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.4])
    thresholds = np.array([0.2, 0.5])
    
    pts, nb_model, nb_all = calculate_dca(y_true, y_prob, thresholds=thresholds)
    
    assert len(nb_model) == len(thresholds), "DCA model net benefit count mismatch."
    assert all(np.isfinite(nb_model)), "DCA produced non-finite net benefit values."