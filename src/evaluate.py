from sklearn.metrics import classification_report
import os

def evaluate_model(model, label_encoder, X_test, y_test, results_dir):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print(report)
