import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_from_source(filepath: str):
    """Loads data from a specified CSV file path."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    return df

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """Splits data into train and test sets, ensuring sensitive features are aligned."""
    if 'creditworthy' not in df.columns or 'location_tier' not in df.columns:
        raise ValueError("DataFrame must contain 'creditworthy' and 'location_tier' columns.")

    X = df.drop(columns=['partner_id', 'creditworthy', 'location_tier'], errors='ignore')
    y = df['creditworthy']
    sensitive_features = df['location_tier']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, sf_train, sf_test