from src.data import load_multimodal_cohort
from src.features import engineer_clinical_features
from src.models import ClinicalRiskPipeline

def run():
    print("==================================================")
    print("  CLINICAL RISK PIPELINE: PROLONGED LOS PREDICTION")
    print("==================================================\n")
    
    print("[1/3] Querying clinical database...")
    raw_df = load_multimodal_cohort()
    print(f"      Loaded {len(raw_df)} admissions.\n")
    
    print("[2/3] Engineering clinical trajectories and indicators...")
    X, y, features = engineer_clinical_features(raw_df)
    print(f"      Feature matrix: {X.shape[1]} covariates.\n")
    
    print("[3/3] Training and evaluating on holdout test set...")
    pipeline = ClinicalRiskPipeline()
    metrics = pipeline.fit_evaluate(X, y, features)
    
    print("\n--- MODEL PERFORMANCE REPORT ---")
    print(f"Holdout ROC-AUC : {metrics['auc']:.3f}\n")
    print("Confusion Matrix (Cutoff = 0.50):")
    print(metrics['confusion_matrix'])
    print("\nStandardized Clinical Odds Ratios:")
    print(metrics['odds_ratios'].to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    run()