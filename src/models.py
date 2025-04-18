import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    # Target column
    target_col = 'pH'
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])

    # Split into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label encode any non-numeric columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_save_path)

    return model, X_test, y_test
