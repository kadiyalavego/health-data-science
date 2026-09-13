import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_prob, save_path: str = "reports/calibration_curve.png"):
    """Evaluates probabilistic reliability against perfect calibration."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="s", color="#1f77b4", label="XGBoost Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives (LOS > 7d)")
    plt.title("Reliability Diagram (Model Calibration)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Calibration plot saved to: {save_path}")


def calculate_dca(y_true, y_prob, thresholds=np.linspace(0.05, 0.80, 50)):
    """
    Computes Clinical Net Benefit across decision threshold probabilities:
    Net Benefit = (True Positives / N) - (False Positives / N) * (p_t / (1 - p_t))
    """
    n = len(y_true)
    prevalence = np.mean(y_true)
    net_benefit_model = []
    net_benefit_all = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit_model.append(nb)
        
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        net_benefit_all.append(nb_all)

    return thresholds, net_benefit_model, net_benefit_all


def plot_decision_curve(y_true, y_prob, save_path: str = "reports/decision_curve_analysis.png"):
    """Renders Decision Curve Analysis to demonstrate clinical utility."""
    thresholds, nb_model, nb_all = calculate_dca(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, nb_model, color="#2ca02c", lw=2, label="XGBoost Strategy")
    plt.plot(thresholds, nb_all, color="#d62728", linestyle="--", label="Intervene on All Patients")
    plt.axhline(0, color="black", linestyle=":", label="Intervene on None")
    
    plt.ylim(-0.05, max(nb_model) + 0.05)
    plt.xlim(0.05, 0.80)
    plt.xlabel("Bedside Clinical Decision Threshold ($p_t$)")
    plt.ylabel("Standardized Net Benefit")
    plt.title("Decision Curve Analysis (Clinical Net Benefit)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Decision Curve plot saved to: {save_path}")