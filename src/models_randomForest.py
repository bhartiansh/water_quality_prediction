import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def train_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    # Handle string entries like '>6.3', '<0.005'
    df.replace(to_replace=r"[<>]([0-9.]+)", value=lambda m: str(m.group(1)), regex=True, inplace=True)

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in dataset.")

    X = df.drop(columns=['label'])
    y = df['label']

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Optional: scale features (not necessary for RF but helps with consistency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save
    joblib.dump((model, le), model_save_path)

    return model, le, X_test, y_test
