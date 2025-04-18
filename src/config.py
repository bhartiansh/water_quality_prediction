import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'water_quality_raw.csv')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed', 'water_quality_cleaned.csv')
THRESHOLDS_PATH = os.path.join(BASE_DIR, 'data', 'thresholds.csv')

MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')
PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
