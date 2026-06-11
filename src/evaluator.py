"""
evaluator.py
============
Full model evaluation:
  - Classification report (Accuracy, Precision, Recall, F1)
  - Confusion matrix heatmap
  - ROC-AUC curve
  - Precision-Recall curve
  - Threshold tuning for clinical use
  - Feature importance via Random Forest coefficients
  - Save all plots to reports/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, json, logging
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score
)

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="muted")
REPORT_DIR = "reports"


def ensure_dir(path=REPORT_DIR):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Classification Report
# ─────────────────────────────────────────────
def print_classification_report(y_true, y_pred):
    print("\n" + "="*55)
    print("  CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(y_true, y_pred,
          target_names=["No Diabetes", "Diabetes"]))


# ─────────────────────────────────────────────
# 2. Confusion Matrix Heatmap
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save=True):
    ensure_dir()
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: No", "Predicted: Yes"],
        yticklabels=["Actual: No",    "Actual: Yes"],
        linewidths=0.5, ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=14)

    # Annotate clinical cost labels
    ax.text(0.5, -0.12,
            f"TN={tn}  FP={fp}  FN={fn}  TP={tp} | "
            f"Sensitivity={tp/(tp+fn):.2f}  Specificity={tn/(tn+fp):.2f}",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")

    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        logger.info(f"Saved → {path}")
    plt.close()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


# ─────────────────────────────────────────────
# 3. ROC Curve
# ─────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob, model_name="XGBoost", save=True):
    ensure_dir()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6366f1", lw=2.5, label=f"{model_name}  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#6366f1")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)",     fontsize=11)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "roc_curve.png")
        fig.savefig(path, dpi=150)
        logger.info(f"Saved → {path}")
    plt.close()
    return round(auc, 4)


# ─────────────────────────────────────────────
# 4. Precision-Recall Curve
# ─────────────────────────────────────────────
def plot_precision_recall_curve(y_true, y_prob, save=True):
    ensure_dir()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#10b981", lw=2.5,
            label=f"Model  (AP = {ap:.3f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1.2,
               label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall (Sensitivity)",  fontsize=11)
    ax.set_ylabel("Precision",             fontsize=11)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "precision_recall_curve.png")
        fig.savefig(path, dpi=150)
        logger.info(f"Saved → {path}")
    plt.close()
    return round(ap, 4)


# ─────────────────────────────────────────────
# 5. Threshold Tuning (clinical priority: minimise FN)
# ─────────────────────────────────────────────
def tune_threshold(y_true, y_prob, min_precision=0.60, save=True):
    """
    Find the lowest threshold where precision >= min_precision.
    In a medical screening context we want high recall (catch diabetics)
    while keeping precision acceptable.
    """
    ensure_dir()
    thresholds = np.arange(0.10, 0.90, 0.01)
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        acc  = accuracy_score(y_true, y_pred_t)
        f1   = f1_score(y_true, y_pred_t, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred_t).ravel()
        tn, fp, fn, tp = cm if len(cm) == 4 else (0, 0, 0, 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rows.append({"threshold": t, "accuracy": acc, "f1": f1,
                     "sensitivity": sens, "specificity": spec, "precision": prec})

    df = pd.DataFrame(rows)

    # Best threshold: max F1 while precision >= min_precision
    viable = df[df["precision"] >= min_precision]
    if viable.empty:
        best_row = df.loc[df["f1"].idxmax()]
    else:
        best_row = viable.loc[viable["f1"].idxmax()]

    best_threshold = round(best_row["threshold"], 2)
    logger.info(f"Optimal threshold: {best_threshold:.2f} "
                f"| F1={best_row['f1']:.3f} "
                f"| Sensitivity={best_row['sensitivity']:.3f} "
                f"| Precision={best_row['precision']:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["threshold"], df["f1"],          label="F1 Score",    color="#6366f1", lw=2)
    ax.plot(df["threshold"], df["sensitivity"], label="Sensitivity", color="#10b981", lw=2)
    ax.plot(df["threshold"], df["specificity"], label="Specificity", color="#f59e0b", lw=2)
    ax.plot(df["threshold"], df["precision"],   label="Precision",   color="#ef4444", lw=2)
    ax.axvline(x=best_threshold, color="gray", linestyle="--", lw=1.5,
               label=f"Best threshold ({best_threshold})")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Score",              fontsize=11)
    ax.set_title("Threshold Tuning",    fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "threshold_tuning.png")
        fig.savefig(path, dpi=150)
        logger.info(f"Saved → {path}")
    plt.close()
    return best_threshold, df


# ─────────────────────────────────────────────
# 6. Feature Importance (from tree-based model)
# ─────────────────────────────────────────────
def plot_feature_importance(pipeline, feature_names: list, save=True):
    ensure_dir()
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        logger.warning("Model has no feature_importances_. Skipping plot.")
        return None

    idx  = np.argsort(importances)
    cols = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(cols)))
    bars = ax.barh(cols, vals, color=colors, edgecolor="none", height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("Feature Importance — XGBoost", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(REPORT_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150)
        logger.info(f"Saved → {path}")
    plt.close()

    importance_dict = dict(zip(feature_names, importances.tolist()))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────
# 7. Full Evaluation Summary
# ─────────────────────────────────────────────
def full_evaluation(pipeline, X_test, y_test, feature_names: list,
                    model_name="XGBoost", threshold=0.5):
    """
    Run all evaluations in one call. Returns a summary dict.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Apply custom threshold if different from default
    if threshold != 0.5:
        y_pred = (y_prob >= threshold).astype(int)
        logger.info(f"Applying custom threshold: {threshold}")

    print_classification_report(y_test, y_pred)

    cm_stats  = plot_confusion_matrix(y_test, y_pred)
    auc       = plot_roc_curve(y_true=y_test, y_prob=y_prob, model_name=model_name)
    ap        = plot_precision_recall_curve(y_test, y_prob)
    best_thr, _ = tune_threshold(y_test, y_prob)
    fi        = plot_feature_importance(pipeline, feature_names)

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)

    summary = {
        "accuracy":   round(accuracy, 4),
        "roc_auc":    auc,
        "avg_precision": ap,
        "f1_positive": round(f1, 4),
        "optimal_threshold": best_thr,
        "confusion_matrix": cm_stats,
        "feature_importance": fi,
    }

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    for k, v in summary.items():
        if k not in ("confusion_matrix", "feature_importance"):
            print(f"  {k:<25}: {v}")
    print("="*55)

    return summary


def save_evaluation_report(summary: dict, path: str = "reports/evaluation_report.json"):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation report saved → {path}")