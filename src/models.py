import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns=['label'])
    y = df['label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump((model, le), model_save_path)
    return model, le, X_test, y_test
