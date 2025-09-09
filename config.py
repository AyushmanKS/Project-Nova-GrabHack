# config.py

# --- File Paths ---
DATA_PATH = './data/grab_partner_data.csv'
MODEL_DIR = './models/'
FINAL_MODEL_NAME = 'production_credit_scorer.joblib'

# --- Feature Engineering ---
# List of features to be used in the model
FEATURES = [
    'weekly_earnings', 'trip_frequency', 'customer_ratings', 'driving_score',
    'earnings_per_trip', 'rating_to_driving_ratio', 'earnings_x_rating',
    'earnings_consistency_std', # NEW
    'rating_trend',             # NEW
    'high_demand_acceptance_rate' # NEW
]

# --- Model Training ---
TARGET_VARIABLE = 'creditworthy'
SENSITIVE_FEATURE = 'location_tier'
TEST_SIZE = 0.25
RANDOM_STATE = 42

# --- Fairness ---
# Define the acceptable threshold for fairness. A real business would decide this.
FAIRNESS_THRESHOLD_EOD = 0.10 # e.g., We accept a maximum of 10% EOD