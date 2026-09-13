import shap
import matplotlib.pyplot as plt
import os

def generate_shap_explanations(model, X_train, X_test, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Global summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Global SHAP summary plot saved to: {summary_path}")
    
    # Local waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    waterfall_path = os.path.join(output_dir, "shap_waterfall_patient_0.png")
    plt.savefig(waterfall_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Local SHAP waterfall plot saved to: {waterfall_path}")
