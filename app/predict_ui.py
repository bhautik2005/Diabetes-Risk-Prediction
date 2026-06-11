import streamlit as st
import pandas as pd
import joblib, shap, numpy as np
import os, sys

# Set path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessor import engineer_features

model = joblib.load('models/best_model.pkl')

st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter patient health indicators to get a risk assessment.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies",        0, 20, 0)
        glucose     = st.number_input("Glucose (mg/dL)",    0, 300, 120)
        bp          = st.number_input("Blood Pressure",      0, 200, 72)
        skin        = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with col2:
        insulin     = st.number_input("Insulin (μU/mL)",    0, 900, 80)
        bmi         = st.number_input("BMI",                0.0, 70.0, 25.0)
        dpf         = st.number_input("Pedigree Function",  0.0, 3.0,  0.3)
        age         = st.number_input("Age",                21, 90, 30)
    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    patient = pd.DataFrame([[pregnancies, glucose, bp, skin,
                              insulin, bmi, dpf, age]],
                columns=['Pregnancies','Glucose','BloodPressure',
                         'SkinThickness','Insulin','BMI',
                         'DiabetesPedigreeFunction','Age'])

    # Engineer features to prevent training-serving skew
    patient_eng = engineer_features(patient)

    prob  = model.predict_proba(patient_eng)[0, 1]
    label = " HIGH RISK 🔴" if prob >= 0.45 else "LOW RISK 🟢"

    st.metric("Diabetes Risk Score", f"{prob:.01%}",)
    # Colored message
    if prob >= 0.45:
       st.error(label)      # Red box
    else:
       st.success(label)
       # Green box
     
     
    st.progress(float(prob))

    # SHAP waterfall for this patient
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    explainer = shap.TreeExplainer(classifier)
    X_t = preprocessor.transform(patient_eng)
    
    # Thread-safe Matplotlib rendering for Streamlit
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(explainer(X_t)[0], show=False)
    st.pyplot(fig)
    plt.close(fig)
