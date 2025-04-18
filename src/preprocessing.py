import pandas as pd

def clean_data(excel_path):
    # Read the xlsm file
    df = pd.read_excel(excel_path, engine='openpyxl')

    # Example cleaning steps
    df.dropna(inplace=True)  # Remove rows with missing values
    df = df.drop(columns=['Water Control Zone', 'Station', 'Dates', 'Sample No'], errors='ignore')  # Drop non-numeric

    # Ensure all column names are stripped of extra whitespace
    df.columns = df.columns.str.strip()

    return df
