import pandas as pd

def check_threshold_violations(df, threshold_path):
    thresholds = pd.read_csv(threshold_path)
    violations = {}
    for _, row in thresholds.iterrows():
        column = row['parameter']
        max_val = row['max_value']
        if column in df.columns:
            violations[column] = (df[column] > max_val).sum()
    return violations
