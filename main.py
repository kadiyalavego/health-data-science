import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from src.data import load_multimodal_cohort
from src.features import engineer_clinical_features
from src.models import ClinicalRiskPipeline, train_xgboost_model
from src.explainability import generate_shap_explanations

def run_pipeline():
    print("\n==========================================")
    print(" Executing Clinical ML & SHAP Pipeline")
    print("==========================================\n")
    
    # 1. Ingestion
    print("Loading multimodal patient cohort...")
    df = load_multimodal_cohort("clinical_data.db")
    
    # 2. Feature Engineering
    X, y, feature_names = engineer_clinical_features(df)
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Baseline Interpretable Model
    print("Fitting baseline logistic regression pipeline...")
    pipeline = ClinicalRiskPipeline()
    metrics = pipeline.fit_evaluate(X, y, feature_names)
    
    # 4. Non-Linear Modeling (XGBoost)
    xgb_model, xgb_auc = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # 5. Model Explainability (SHAP)
    print("\nGenerating bedside SHAP feature attribution reports...")
    generate_shap_explanations(xgb_model, X_train, X_test)
    
    print("\nPipeline execution complete. Reports saved to 'reports/' directory.\n")

if __name__ == "__main__":
    run_pipeline()