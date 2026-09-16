import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
import xgboost as xgb


class ClinicalRiskPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit standard scaler and model to input features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def fit_evaluate(self, X: pd.DataFrame, y: pd.Series, feature_names=None):
        """
        Splits data 80/20, fits model, evaluates discrimination and calibration,
        and extracts clinical Odds Ratios.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.fit(X_train, y_train)
        y_prob = self.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        cols = feature_names if feature_names is not None else X.columns
        odds_ratios = pd.DataFrame({
            'Feature': cols,
            'Odds_Ratio': np.exp(self.model.coef_[0])
        })

        return {
            'auc': auc,
            'brier_score': brier,
            'confusion_matrix': cm,
            'odds_ratios': odds_ratios,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train gradient-boosted tree model for nonlinear interactions."""
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return clf, y_prob