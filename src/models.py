import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model(processed_path, model_save_path):
    df = pd.read_csv(processed_path)
    
    # Drop non-numeric or non-feature columns
    X = df.drop(columns=['Water Control Zone', 'Station', 'Dates', 'Sample No', 'pH'])
    y = df['pH']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_save_path)
    return model, X_test, y_test
