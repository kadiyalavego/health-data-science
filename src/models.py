import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, brier_score_loss
import xgboost as xgb


class ClinicalRiskPipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)

    def fit_evaluate(self, X: pd.DataFrame, y: pd.Series, feature_names: list) -> dict:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)
        
        odds_ratios = np.exp(self.model.coef_[0])
        or_df = pd.DataFrame({
            'Feature': feature_names,
            'Odds_Ratio': odds_ratios
        }).sort_values(by='Odds_Ratio', ascending=False)
        
        print("\n--- BASELINE LOGISTIC REGRESSION ---")
        print(f"Cohort ROC-AUC: {auc:.4f}")
        print("\nTop Clinical Odds Ratios:")
        print(or_df.head(5).to_string(index=False))
        
        return {
            'auc': auc,
            'confusion_matrix': cm,
            'odds_ratios': or_df
        }


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Fits a regularized XGBoost classifier to capture non-linear interactions
    and outputs clinical evaluation metrics (AUC-ROC, Brier calibration score).
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)
    
    print("\n--- XGBOOST MODEL EVALUATION ---")
    print(f"Test Set ROC-AUC    : {auc:.4f}")
    print(f"Test Set Brier Score: {brier:.4f}")
    
    return model, auc