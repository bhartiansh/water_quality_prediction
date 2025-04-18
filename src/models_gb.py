# src/models_gb.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def train_gb_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    drop_cols = [col for col in ['Water Control Zone', 'Station', 'Dates', 'Sample No'] if col in df.columns]
    df = df.drop(columns=drop_cols)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    joblib.dump((model, le), model_save_path)

    return model, le, X_test, y_test

def evaluate_model(model, label_encoder, X_test, y_test, output_path):
    y_pred = model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    report = classification_report(y_test_labels, y_pred_labels, digits=2)
    print(report)

    with open(output_path, "w") as f:
        f.write(report)
