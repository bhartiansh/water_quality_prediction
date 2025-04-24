import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_for_model(dfs):
    # Combine all datasets into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing labels (shouldn’t happen but just in case)
    combined_df.dropna(subset=['label'], inplace=True)

    # Encode labels
    y = combined_df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Drop label column and keep only numeric features
    X = combined_df.drop(columns=['label'])
    X = X.select_dtypes(include=[np.number])

    # Impute missing values and scale
    X_imputed = SimpleImputer(strategy='mean').fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42), le
