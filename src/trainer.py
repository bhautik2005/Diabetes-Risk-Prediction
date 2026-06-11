"""
trainer.py
==========
Trains multiple classifier candidates, performs cross-validation,
hyperparameter tuning, and selects the best model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from xgboost               import XGBClassifier
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV
)
import joblib, os, json, logging
from datetime import datetime

logger = logging.getLogger(__name__)

CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM":                 SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost":             XGBClassifier(
                               n_estimators=200, eval_metric="logloss",
                               random_state=42, verbosity=0
                           ),
}

XGBOOST_PARAM_GRID = {
    "classifier__n_estimators":     [100, 200, 300],
    "classifier__max_depth":        [3, 5, 7],
    "classifier__learning_rate":    [0.01, 0.05, 0.1],
    "classifier__subsample":        [0.8, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
    "classifier__scale_pos_weight": [1, 1.86],   # handles 65/35 class imbalance
}


def compare_models(preprocessor, X_train, y_train, cv_folds=5):
    """
    Cross-validate all candidate models and print a comparison table.

    Returns
    -------
    results : dict  {model_name: {"mean_auc": float, "std_auc": float}}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'Model':<25} {'Mean AUC':>10} {'Std':>8}")
    print("-" * 45)

    for name, clf in CANDIDATE_MODELS.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf)
        ])
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {"mean_auc": scores.mean(), "std_auc": scores.std()}
        print(f"{name:<25} {scores.mean():>10.4f} {scores.std():>8.4f}")

    best_name = max(results, key=lambda k: results[k]["mean_auc"])
    logger.info(f"\nBest candidate: {best_name} (AUC={results[best_name]['mean_auc']:.4f})")
    return results, best_name


def tune_xgboost(preprocessor, X_train, y_train, n_iter=50):
    """
    RandomizedSearchCV to tune XGBoost hyperparameters.
    Returns the best fitted pipeline.
    """
    logger.info(f"Tuning XGBoost with {n_iter} iterations …")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0))
    ])

    search = RandomizedSearchCV(
        base_pipe, XGBOOST_PARAM_GRID,
        n_iter=n_iter, cv=cv, scoring="roc_auc",
        n_jobs=-1, verbose=1, random_state=42
    )
    search.fit(X_train, y_train)

    logger.info(f"Best CV AUC: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def save_model(pipeline, params: dict, cv_auc: float,
               feature_names: list, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    meta_path  = os.path.join(model_dir, "model_metadata.json")

    joblib.dump(pipeline, model_path)
    metadata = {
        "model_version":  "1.0.0",
        "training_date":  datetime.now().isoformat(),
        "algorithm":      "XGBoost (RandomizedSearchCV tuned)",
        "feature_names":  feature_names,
        "cv_roc_auc":     round(cv_auc, 4),
        "best_params":    params,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Metadata saved → {meta_path}")
    return model_path


def load_model(model_path: str = "models/best_model.pkl"):
    return joblib.load(model_path)