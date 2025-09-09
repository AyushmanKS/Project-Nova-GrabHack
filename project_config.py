DATA_PATH = './data/grab_partner_data.csv'
MODEL_DIR = './models/'
FINAL_MODEL_NAME = 'production_credit_scorer.joblib'
SCALER_NAME = 'scaler.joblib'

# Feature Engineering
# Defines the final feature set for the model.
FEATURES = [
    'weekly_earnings', 'trip_frequency', 'customer_ratings', 'driving_score',
    'earnings_per_trip', 'rating_to_driving_ratio', 'earnings_x_rating',
    'earnings_consistency_std',
    'rating_trend',
    'high_demand_acceptance_rate'
]

# Model Training
TARGET_VARIABLE = 'creditworthy'
SENSITIVE_FEATURE = 'location_tier'
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Fairness
# The business-defined acceptable threshold for Equalized Odds Difference.
FAIRNESS_THRESHOLD_EOD = 0.10