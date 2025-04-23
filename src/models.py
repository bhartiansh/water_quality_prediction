import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def preprocess_for_model(df):
    df.replace(to_replace=r"<([0-9.]+)", value=lambda m: str(m.group(1)), regex=True, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    X = df.drop(columns=['label'])
    y = df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = X.select_dtypes(include=[np.number])
    X_imputed = SimpleImputer(strategy='mean').fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imputed)
    return train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42), le

def train_model(X_train, y_train, model_type):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientDescent': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    }

    model = models[model_type]
    model.fit(X_train, y_train)
    return model
