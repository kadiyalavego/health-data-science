import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

class ClinicalRiskPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=random_state)
        self.features = []

    def fit_evaluate(self, X: pd.DataFrame, y: pd.Series, feature_names: list) -> dict:
        self.features = feature_names
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )
        
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        self.model.fit(X_train_s, y_train)
        y_prob = self.model.predict_proba(X_test_s)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, (y_prob >= 0.50).astype(int))
        
        or_df = pd.DataFrame({
            'Feature': self.features,
            'Odds_Ratio': np.exp(self.model.coef_[0]),
            'Beta': self.model.coef_[0]
        }).sort_values(by='Odds_Ratio', ascending=False)
        
        return {
            'auc': auc,
            'confusion_matrix': cm,
            'odds_ratios': or_df
        }