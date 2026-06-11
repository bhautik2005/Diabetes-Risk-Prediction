"""
predictor.py
============
Clean inference class — wraps the serialised pipeline for production use.
Accepts raw patient data (dict or DataFrame) and returns risk score + label.
"""

import pandas as pd
import numpy as np
import joblib, json, logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

RISK_LEVELS = {
    (0.00, 0.30): ("LOW",    "🟢", "Low risk of diabetes."),
    (0.30, 0.55): ("MEDIUM", "🟡", "Borderline risk. Lifestyle review recommended."),
    (0.55, 0.75): ("HIGH",   "🟠", "High risk. Clinical assessment advised."),
    (0.75, 1.01): ("CRITICAL","🔴","Very high risk. Immediate clinical referral recommended."),
}


from preprocessor import engineer_features

class DiabetesPredictor:
    """
    Wrapper around the trained sklearn Pipeline for clean inference.

    Usage
    -----
    predictor = DiabetesPredictor("models/best_model.pkl")
    result    = predictor.predict_single(glucose=148, bmi=33.6, age=50, ...)
    print(result)
    """

    def __init__(self, model_path: str = "models/best_model.pkl",
                 meta_path: str  = "models/model_metadata.json",
                 threshold: float = 0.45):
        self.pipeline  = joblib.load(model_path)
        self.threshold = threshold

        try:
            with open(meta_path) as f:
                self.metadata = json.load(f)
            logger.info(f"Model loaded | version={self.metadata['model_version']} "
                        f"| AUC={self.metadata.get('cv_roc_auc','N/A')}")
        except FileNotFoundError:
            self.metadata = {}
            logger.warning("model_metadata.json not found — proceeding without it.")

    def _risk_label(self, prob: float):
        for (lo, hi), (level, icon, advice) in RISK_LEVELS.items():
            if lo <= prob < hi:
                return level, icon, advice
        return "UNKNOWN", "❓", ""

    def predict_single(self, pregnancies=0, glucose=120, blood_pressure=72,
                       skin_thickness=20, insulin=80, bmi=25.0,
                       diabetes_pedigree=0.3, age=30) -> dict:
        """
        Predict diabetes risk for a single patient using keyword arguments.

        Returns
        -------
        dict with keys: probability, risk_level, icon, advice, prediction (0/1)
        """
        patient = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]], columns=FEATURE_COLS)
        return self._predict_df(patient)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for a DataFrame of patients.
        Adds columns: risk_prob, risk_level, prediction.
        """
        results = self._predict_df(df[FEATURE_COLS])
        out = df.copy()
        out["risk_prob"]   = [r["probability"]  for r in results]
        out["risk_level"]  = [r["risk_level"]   for r in results]
        out["prediction"]  = [r["prediction"]   for r in results]
        return out

    def _predict_df(self, X: pd.DataFrame) -> list:
        X_eng = engineer_features(X)
        probs      = self.pipeline.predict_proba(X_eng)[:, 1]
        predictions = (probs >= self.threshold).astype(int)
        results = []
        for prob, pred in zip(probs, predictions):
            level, icon, advice = self._risk_label(prob)
            results.append({
                "probability": round(float(prob), 4),
                "prediction":  int(pred),
                "risk_level":  level,
                "icon":        icon,
                "advice":      advice,
            })
        return results

    def explain_patient(self, patient_dict: dict) -> None:
        """Print a human-readable risk summary for one patient."""
        result = self.predict_single(**patient_dict)
        print("\n" + "-"*45)
        try:
            print(f"  {result['icon']}  DIABETES RISK ASSESSMENT")
        except UnicodeEncodeError:
            print(f"  [{result['risk_level']}]  DIABETES RISK ASSESSMENT")
        print("-"*45)
        print(f"  Probability : {result['probability']:.1%}")
        print(f"  Risk Level  : {result['risk_level']}")
        print(f"  Advice      : {result['advice']}")
        print(f"  Prediction  : {'Diabetic' if result['prediction'] else 'Not Diabetic'}")
        print("-"*45)


# ── Quick CLI test ────────────────────────────────────
if __name__ == "__main__":
    p = DiabetesPredictor()

    # Sample from the dataset (first row: 6 pregnancies, glucose 148, …)
    p.explain_patient({
        "pregnancies":        6,
        "glucose":            148,
        "blood_pressure":     72,
        "skin_thickness":     35,
        "insulin":            0,
        "bmi":                33.6,
        "diabetes_pedigree":  0.627,
        "age":                50,
    })