import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Step 1: Replace biologically impossible zeros with NaN
ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def replace_invalid_zeros(df):
    df = df.copy()
    for col in ZERO_INVALID_COLS:
        df[col] = df[col].replace(0, np.nan)
    return df

# Step 2: Feature engineering (clinical domain knowledge)
def engineer_features(df):
    df = df.copy()
    # BMI categories (WHO classification)
    df['BMI_Category'] = pd.cut(df['BMI'],
        bins=[0,18.5,25,30,100],
        labels=[0,1,2,3]).astype(float)

    # Glucose-to-Insulin ratio (insulin resistance proxy)
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

    # Age group buckets
    df['Age_Group'] = pd.cut(df['Age'],
        bins=[20,30,40,50,100],
        labels=[0,1,2,3]).astype(float)

    # Pregnancy history flag
    df['High_Preg'] = (df['Pregnancies'] > 5).astype(int)
    return df

# Step 3: Build sklearn Pipeline (prevents leakage)
def build_preprocessor(feature_names=None):
    if feature_names is None:
        numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure',
                            'SkinThickness', 'Insulin', 'BMI',
                            'DiabetesPedigreeFunction', 'Age']
    else:
        numeric_features = list(feature_names)
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),  # KNN > median for clinical
        ('scaler',  RobustScaler()),              # robust to outliers
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
    return preprocessor

# Step 4: Stratified split to preserve class ratio
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=y
    )