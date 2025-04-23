import pandas as pd

def clean_and_label_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    
    # Drop metadata columns
    df = df.drop(columns=['Water Control Zone', 'Station', 'Dates', 'Sample No'], errors='ignore')
    df.columns = df.columns.str.strip()

    # Ensure relevant columns are numeric
    cols_to_use = ['Dissolved Oxygen (mg/L)', 'Turbidity (NTU)', 'E. coli (cfu/100mL)']
    for col in cols_to_use:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing key features
    df.dropna(subset=cols_to_use, inplace=True)

    # Label assignment logic
    def assign_label(row):
        do = row['Dissolved Oxygen (mg/L)']
        turb = row['Turbidity (NTU)']
        ecoli = row['E. coli (cfu/100mL)']
        
        if do > 6 and turb < 5 and ecoli < 100:
            return 'Good'
        elif do >= 4 and turb <= 10 and ecoli <= 1000:
            return 'Moderate'
        else:
            return 'Poor'

    df['label'] = df.apply(assign_label, axis=1)
    
    return df
