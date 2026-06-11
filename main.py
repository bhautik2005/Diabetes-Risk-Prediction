"""
main.py
=======
Master orchestration script — runs the complete end-to-end pipeline:
  1. Load data
  2. Engineer features + split
  3. Compare models (cross-validation)
  4. Tune XGBoost
  5. Evaluate on test set
  6. SHAP explainability
  7. Save model + reports

Run with:  python main.py
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader      import load_data, get_feature_target
from preprocessor     import engineer_features, build_preprocessor, split_data
from trainer          import compare_models, tune_xgboost, save_model
from evaluator        import full_evaluation, save_evaluation_report
from explainability   import run_full_explainability


def main():
    print("\n" + "="*60)
    print("  DIABETES RISK PREDICTION — FULL PIPELINE")
    print("="*60 + "\n")

    # ── STEP 1: Load & validate ──────────────────────────────────
    logger.info("STEP 1 | Loading and validating dataset …")
    df = load_data("data/raw/diabetes.csv")

    # ── STEP 2: Feature engineering ─────────────────────────────
    logger.info("STEP 2 | Engineering features …")
    df_engineered = engineer_features(df)
    X, y = get_feature_target(df_engineered)

    # Use all columns as features (original + engineered)
    feature_names = list(X.columns)
    logger.info(f"Total features: {len(feature_names)} → {feature_names}")

    # ── STEP 3: Train / test split ───────────────────────────────
    logger.info("STEP 3 | Splitting data (stratified 80/20) …")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── STEP 4: Compare baseline models ─────────────────────────
    logger.info("STEP 4 | Cross-validating candidate models …")
    preprocessor = build_preprocessor(feature_names)
    results, best_name = compare_models(preprocessor, X_train, y_train)

    # ── STEP 5: Tune best model (XGBoost) ───────────────────────
    logger.info("STEP 5 | Tuning XGBoost hyperparameters …")
    preprocessor = build_preprocessor(feature_names)   # fresh unfitted copy
    best_pipeline, best_params, cv_auc = tune_xgboost(
        preprocessor, X_train, y_train, n_iter=30      # increase to 50+ for production
    )

    # ── STEP 6: Final fit on full training set ───────────────────
    logger.info("STEP 6 | Final model fit on full training set …")
    best_pipeline.fit(X_train, y_train)

    # ── STEP 7: Evaluate on held-out test set ───────────────────
    logger.info("STEP 7 | Evaluating on test set …")
    summary = full_evaluation(
        best_pipeline, X_test, y_test,
        feature_names=feature_names,
        model_name="XGBoost (tuned)",
        threshold=0.45          # clinical threshold: prefer sensitivity
    )

    # ── STEP 8: SHAP explainability ─────────────────────────────
    logger.info("STEP 8 | Running SHAP explainability …")
    run_full_explainability(best_pipeline, X_test, feature_names)

    # ── STEP 9: Save model + metadata ───────────────────────────
    logger.info("STEP 9 | Saving model and reports …")
    summary["cv_roc_auc"]    = round(cv_auc, 4)
    summary["model_name"]    = best_name
    summary["feature_names"] = feature_names

    save_model(best_pipeline, best_params, cv_auc, feature_names)
    save_evaluation_report(summary)

    print("\n" + "="*60)
    print("  [SUCCESS]  PIPELINE COMPLETE")
    print(f"  Model      : {best_name}")
    print(f"  Test AUC   : {summary['roc_auc']}")
    print(f"  Test F1    : {summary['f1_positive']}")
    print(f"  Accuracy   : {summary['accuracy']}")
    print(f"  Threshold  : {summary['optimal_threshold']}")
    print(f"  Artifacts  -> models/  &  reports/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()