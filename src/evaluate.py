import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, results_dir):
    """Evaluate the model and save visual results to the results_dir."""
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Predict
    predictions = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("Model Evaluation Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save metrics to CSV
    metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    pd.DataFrame({
        'Metric': ['R2', 'MAE', 'MSE', 'RMSE'],
        'Value': [r2, mae, mse, rmse]
    }).to_csv(metrics_path, index=False)

    # Plot 1: Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6, edgecolors='k')
    plt.xlabel("Actual pH")
    plt.ylabel("Predicted pH")
    plt.title("Actual vs Predicted pH")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))
    plt.close()

    # Plot 2: Residuals
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'residuals_distribution.png'))
    plt.close()
