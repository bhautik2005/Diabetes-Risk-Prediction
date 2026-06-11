"""
data_loader.py
==============
Loads and validates the Pima Indians Diabetes dataset.
Handles zero-imputation flagging and basic schema checks.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Columns where 0 is biologically impossible → treat as missing
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

EXPECTED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

FEATURE_COLS = [c for c in EXPECTED_COLUMNS if c != "Outcome"]
TARGET_COL   = "Outcome"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV, validate schema, and replace invalid zeros with NaN.

    Parameters
    ----------
    filepath : str
        Path to diabetes.csv

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with NaN markers for impossible zeros.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns from {filepath}")

    # --- Schema validation ---
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # --- Replace biologically impossible zeros ---
    zero_counts = {}
    for col in ZERO_INVALID_COLS:
        n_zeros = (df[col] == 0).sum()
        if n_zeros > 0:
            zero_counts[col] = n_zeros
            df[col] = df[col].replace(0, np.nan)

    if zero_counts:
        logger.warning("Replaced impossible zeros with NaN:")
        for col, n in zero_counts.items():
            logger.warning(f"   {col}: {n} zeros ({n/len(df)*100:.1f}%)")

    # --- Class balance report ---
    balance = df[TARGET_COL].value_counts()
    logger.info(f"Class balance → Negative (0): {balance[0]} | Positive (1): {balance[1]}")
    logger.info(f"Positive rate: {balance[1]/len(df)*100:.1f}%")

    # --- Missing value summary ---
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Missing values after zero replacement:")
        for col, n in missing.items():
            logger.info(f"   {col}: {n} ({n/len(df)*100:.1f}%)")

    return df


def get_feature_target(df: pd.DataFrame):
    """
    Split DataFrame into features (X) and target (y).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target: {TARGET_COL} | Shape: {X.shape}")
    return X, y


def save_processed(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")


if __name__ == "__main__":
    df = load_data("data/raw/diabetes.csv")
    X, y = get_feature_target(df)
    print(df.describe().round(2))