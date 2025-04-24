import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train, model_type):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SGD': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    }

    model = models[model_type]
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='f1_macro')
    print(f"[{model_type}] Average F1 Score (CV): {cv_scores.mean():.4f}")

    model.fit(X_train_res, y_train_res)
    joblib.dump(model, f"{model_type}_model.pkl")
    return model
