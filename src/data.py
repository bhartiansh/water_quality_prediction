def preprocess_data(input_path, output_path):
    import pandas as pd

    df = pd.read_excel(input_path)

    # Drop irrelevant columns
    drop_cols = ['Water Control Zone', 'Station', 'Dates', 'Sample No']
    df = df.drop(columns=drop_cols)

    # Replace '<' values with float approximations
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('<', '', regex=False)
            try:
                df[col] = df[col].astype(float)
            except:
                pass  # skip if not convertible

    # Encode 'Depth' (still categorical)
    df['Depth'] = df['Depth'].astype(str)
    depth_mapping = {label: idx for idx, label in enumerate(df['Depth'].unique())}
    df['Depth'] = df['Depth'].map(depth_mapping)

    # Label logic
    def classify_water(row):
        pH = row['pH']
        do = row['Dissolved Oxygen (mg/L)']
        if 6.5 <= pH <= 8.5 and do > 5:
            return 'drinkable'
        elif 6.0 <= pH <= 9.0 and do > 3:
            return 'good for marine life'
        else:
            return 'non consumable'

    df['label'] = df.apply(classify_water, axis=1)

    # Drop columns used for classification
    df = df.drop(columns=['pH', 'Dissolved Oxygen (mg/L)'])

    # Save cleaned data
    df.to_csv(output_path, index=False)
