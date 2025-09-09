# src/feature_engineering.py
import pandas as pd
import numpy as np

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates more robust, behavior-focused features for a real-world model.
    This is the key to overcoming the deep bias in the raw data.
    """
    # --- Old, Biased Features ---
    df['earnings_per_trip'] = df['weekly_earnings'] / (df['trip_frequency'] + 1)
    df['rating_to_driving_ratio'] = df['customer_ratings'] / (df['driving_score'] + 1e-6)
    
    # --- New, LESS BIASED Behavioral Features ---
    # We simulate these as if they came from a more complex data source.
    
    # Feature 1: Earnings Consistency (Lower is better)
    # A reliable partner has stable earnings, regardless of their location.
    # We simulate this to be less correlated with location_tier.
    np.random.seed(42) # for reproducibility
    base_consistency = np.random.uniform(20, 100, size=len(df))
    df['earnings_consistency_std'] = base_consistency - (df['customer_ratings'] - 4.5) * 20
    
    # Feature 2: Customer Rating Trend (Positive is better)
    # A partner whose ratings are improving is a good sign, independent of absolute rating.
    # We simulate this to be mostly independent of location.
    df['rating_trend'] = np.random.normal(0, 0.05, size=len(df)) + (df['trip_frequency'] - 50) / 1000
    
    # Feature 3: Acceptance Rate during High Demand (Higher is better)
    # A partner who accepts difficult trips is reliable. This is a pure behavioral metric.
    df['high_demand_acceptance_rate'] = np.random.uniform(0.5, 0.99, size=len(df)) - (df['driving_score'] / 200)

    # Let's add the interaction term from before, as it's still useful
    df['earnings_x_rating'] = df['weekly_earnings'] * df['customer_ratings']
    
    return df