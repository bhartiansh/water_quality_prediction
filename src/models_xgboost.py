import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_xgboost_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    # Clean unusual symbols like '>6.3', '<0.005'
    df.replace(to_replace=r"[<>](\d+\.?\d*)", value=lambda m: m.group(1), regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in dataset.")

    X = df.drop(columns=['label'])
    y = df['label']

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Safety check
    if len(np.unique(y_encoded)) < 2:
        raise ValueError("Target 'y' must contain at least two classes.")

    # Numeric features only
    X = X.select_dtypes(include=[np.number])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train_res, y_train_res)

    # Save model and encoder
    joblib.dump((model, le), model_save_path)

    return model, le, X_test, y_test
