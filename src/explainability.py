"""
explainability.py
=================
SHAP-based explainability for the trained pipeline.
  - Global: Summary bar plot + beeswarm plot
  - Local : Waterfall plot for individual patients
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap, os, logging

logger = logging.getLogger(__name__)
REPORT_DIR = "reports"


def get_shap_values(pipeline, X_data: pd.DataFrame, feature_names: list):
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    X_data        : raw feature DataFrame (will be transformed internally)
    feature_names : list of feature column names

    Returns
    -------
    shap_values  : np.ndarray
    X_transformed: np.ndarray (preprocessed features)
    explainer    : shap.TreeExplainer
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X_data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    logger.info(f"SHAP values computed for {X_transformed.shape[0]} samples")
    return shap_values, X_transformed, explainer


def plot_shap_summary(shap_values, X_transformed, feature_names: list, save=True):
    """Global SHAP summary beeswarm plot — shows direction and magnitude."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        show=False, plot_size=None
    )
    plt.title("SHAP Summary — Global Feature Impact", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    plt.close()


def plot_shap_bar(shap_values, feature_names: list, save=True):
    """Bar chart of mean absolute SHAP values — easy to present to stakeholders."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx  = np.argsort(mean_abs)
    cols = [feature_names[i] for i in idx]
    vals = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(cols)))
    bars = ax.barh(cols, vals, color=colors, edgecolor="none", height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("SHAP Feature Importance (Mean Absolute Impact)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "shap_bar.png")
        fig.savefig(path, dpi=150)
        logger.info(f"Saved → {path}")
    plt.close()


def explain_patient(pipeline, patient_df: pd.DataFrame, feature_names: list,
                    patient_index: int = 0, save=True):
    """
    Generate a SHAP waterfall plot for a single patient.
    Shows which features pushed risk UP or DOWN for that individual.

    Parameters
    ----------
    pipeline      : fitted pipeline
    patient_df    : DataFrame with one or more patient rows
    patient_index : which row to explain (default 0)
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    X_t = preprocessor.transform(patient_df)
    explainer = shap.TreeExplainer(classifier)
    shap_explanation = explainer(X_t)

    prob = pipeline.predict_proba(patient_df)[0, 1]
    risk = "HIGH RISK" if prob >= 0.45 else "LOW RISK"
    logger.info(f"Patient {patient_index}: {risk} — probability = {prob:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_explanation[0], show=False)
    plt.title(f"Patient {patient_index} Explanation — {risk} ({prob:.1%})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, f"shap_patient_{patient_index}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved → {path}")
    plt.close()
    return prob, risk


def run_full_explainability(pipeline, X_test: pd.DataFrame, feature_names: list):
    """Run all SHAP plots in one call."""
    logger.info("Running SHAP explainability analysis …")
    shap_values, X_transformed, explainer = get_shap_values(
        pipeline, X_test, feature_names
    )
    plot_shap_summary(shap_values, X_transformed, feature_names)
    plot_shap_bar(shap_values, feature_names)

    # Explain first 3 patients as examples
    for i in range(min(3, len(X_test))):
        explain_patient(pipeline, X_test.iloc[[i]], feature_names, patient_index=i)

    logger.info("SHAP analysis complete. All plots saved to reports/")
    return shap_values