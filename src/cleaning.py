import pandas as pd

def clean_and_label_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    # Drop metadata
    df = df.drop(columns=['Water Control Zone', 'Station', 'Dates', 'Sample No'], errors='ignore')

    cols_needed = ['Dissolved Oxygen (mg/L)', 'Turbidity (NTU)', 'E. coli (cfu/100mL)']
    missing_cols = [col for col in cols_needed if col not in df.columns]

    if missing_cols:
        print(f"[Missing Columns] {missing_cols} in {path}")
        return pd.DataFrame()

    for col in cols_needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    original_len = len(df)
    df.dropna(subset=cols_needed, inplace=True)
    cleaned_len = len(df)
    print(f"[Cleaned] {path} — Dropped {original_len - cleaned_len} rows with missing key features")

    def assign_label(row):
        do = row['Dissolved Oxygen (mg/L)']
        turb = row['Turbidity (NTU)']
        ecoli = row['E. coli (cfu/100mL)']

        if do > 6 and turb < 5 and ecoli < 100:
            return 'Drinkable'
        elif do >= 4 and turb <= 10 and ecoli <= 1000:
            return 'Good for Marine Life'
        else:
            return 'Non-Drinkable'

    df['label'] = df.apply(assign_label, axis=1)
    return df
