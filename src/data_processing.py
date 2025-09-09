# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CORRECTED FUNCTION ---
def load_data_from_source(filepath: str):
    """
    Loads data from a specified CSV file path.
    In a real-world scenario, this could connect to a database,
    but for now, it reads from the provided path.
    """
    # The function now uses the 'filepath' argument instead of a hardcoded path.
    # This makes the function reusable and fixes the FileNotFoundError.
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    return df

def split_data(df: pd.DataFrame):
    """Splits data into train and test sets, ensuring sensitive features are aligned."""
    # We will assume these columns exist and handle them in the training script
    # This makes the function more generic
    
    # Check if target and sensitive feature are in the dataframe
    if 'creditworthy' not in df.columns or 'location_tier' not in df.columns:
        raise ValueError("DataFrame must contain 'creditworthy' and 'location_tier' columns.")

    X = df.drop(columns=['partner_id', 'creditworthy', 'location_tier'], errors='ignore')
    y = df['creditworthy']
    sensitive_features = df['location_tier']
    
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, sf_train, sf_test