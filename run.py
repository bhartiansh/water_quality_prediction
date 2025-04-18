import os
from src import preprocessing, models, evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
raw_data_path = os.path.join(BASE_DIR, 'data', 'raw', 'water_quality_raw.csv')
processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'water_quality_cleaned.csv')
model_path = os.path.join(BASE_DIR, 'results', 'reports', 'rf_model.pkl')
results_dir = os.path.join(BASE_DIR, 'results')

def main():
    # Load and clean
    df = preprocessing.load_data(raw_data_path)
    df_cleaned = preprocessing.clean_data(df)
    preprocessing.save_processed_data(df_cleaned, processed_path)

    # Train model
    model, X_test, y_test = models.train_model(processed_path, model_path)

    # Evaluate
    mse, r2 = evaluate.evaluate_model(model, X_test, y_test, results_dir)
    print(f"✅ Evaluation complete - MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    main()
