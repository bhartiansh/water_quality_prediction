# src/models_gd.py

import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def train_gd_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    # Replace strings like '<0.005' with numeric part
    df.replace(to_replace=r"<([0-9.]+)", value=lambda m: str(m.group(1)), regex=True, inplace=True)

    # Convert to numeric wherever possible
    df = df.apply(pd.to_numeric, errors='ignore')

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in dataset.")

    X = df.drop(columns=['label'])
    y = df['label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Keep numeric features
    X = X.select_dtypes(include=[np.number])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train Gradient Descent Classifier (Logistic Regression loss)
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train_res, y_train_res)

    # Save model and label encoder
    joblib.dump((model, le), model_save_path)

    return model, le, X_test, y_test
