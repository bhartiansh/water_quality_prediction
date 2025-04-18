import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna()
    df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    return df

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
