import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def train_logistic_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    # Convert non-numeric "<0.005" to numeric values
    df.replace(to_replace=r"<([0-9.]+)", value=lambda m: str(m.group(1)), regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')

    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column in dataset.")

    X = df.drop(columns=['label'])
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X = X.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_res, y_train_res)

    joblib.dump((model, le), model_save_path)
    return model, le, X_test, y_test
