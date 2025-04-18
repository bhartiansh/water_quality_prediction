from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test, results_dir):
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save report
    with open(os.path.join(results_dir, 'reports', 'evaluation.txt'), 'w') as f:
        f.write(f"MSE: {mse:.3f}\nR²: {r2:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel('Actual pH')
    plt.ylabel('Predicted pH')
    plt.title('Actual vs Predicted pH')
    plt.savefig(os.path.join(results_dir, 'figures', 'predicted_vs_actual.png'))
    plt.close()

    return mse, r2
