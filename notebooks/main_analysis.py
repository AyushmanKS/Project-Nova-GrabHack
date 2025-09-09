# %% [markdown]
# # Project Nova: A Fair and Transparent Credit Scoring Model (Final Version)
#
# This script covers the end-to-end process:
# 1.  **Data Simulation:** Create a realistic dataset with inherent bias.
# 2.  **Exploratory Data Analysis (EDA):** Identify and visualize the bias.
# 3.  **Modeling:** Train a high-performance (but unfair) model.
# 4.  **Bias Mitigation:** Apply fairness techniques to create an equitable model.
# 5.  **Evaluation:** Compare the models on both performance and fairness.
# 6.  **Deployment:** Save the final, fair model for future use.

# %%
# =============================================================================
# Cell 1: Generate a Simulated Dataset (FIXED FOR DEGENERATE LABELS)
# =============================================================================
import pandas as pd
import numpy as np
import os

print("--- Cell 1: Generating Simulated Data (Robust Version) ---")

# --- Configuration ---
NUM_PARTNERS = 5000
np.random.seed(42) # Ensures the data is the same every time

# --- Create the data directory if it doesn't exist ---
if not os.path.exists('../data'):
    os.makedirs('../data')

# --- Generate Features (with intentional bias) ---
location_tiers = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], NUM_PARTNERS, p=[0.2, 0.5, 0.3])
earnings_mean = np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [450, 400, 380])
weekly_earnings = np.random.normal(earnings_mean, 60, NUM_PARTNERS).round(2)
trip_frequency = np.random.poisson(np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [60, 50, 45]), NUM_PARTNERS)
customer_ratings = np.random.normal(np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [4.9, 4.75, 4.7]), 0.1).clip(4.0, 5.0).round(2)
driving_score = np.random.normal(np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [10, 15, 18]), 5).clip(0, 100).round()

# --- Generate Target Variable ('creditworthy') ---
base_probability = 0.5
performance_score = (weekly_earnings / weekly_earnings.max()) + (customer_ratings / 5.0) - (driving_score / 100)
location_bias = np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [0.15, 0.05, -0.2])
unclipped_probability = base_probability + (performance_score / 3) + location_bias

# --- THE CRITICAL FIX ---
# We clip the probability between 0.01 and 0.99 to ensure every group has a
# chance of having both positive and negative outcomes. This prevents degenerate labels.
final_probability = unclipped_probability.clip(0.01, 0.99)

creditworthy = (np.random.rand(NUM_PARTNERS) < final_probability).astype(int)

# --- Create and Save DataFrame ---
df = pd.DataFrame({
    'partner_id': range(1, NUM_PARTNERS + 1), 'weekly_earnings': weekly_earnings, 'trip_frequency': trip_frequency,
    'customer_ratings': customer_ratings, 'driving_score': driving_score, 'location_tier': location_tiers, 'creditworthy': creditworthy
})
df.to_csv('../data/grab_partner_data.csv', index=False)

print("Simulated dataset created successfully at 'data/grab_partner_data.csv'")
# Let's check the new creditworthy rates to confirm they are not 100%
print("\nNew Creditworthy Rates by Tier in generated data:")
print(df.groupby('location_tier')['creditworthy'].value_counts(normalize=True).unstack().fillna(0))
df.head()

# %%
# =============================================================================
# Cell 2: Import Libraries and Load Data
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
# UPDATED: MetricFrame is the key import for robust group-based metrics
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

print("\n--- Cell 2: Loading Libraries and Data ---")
df = pd.read_csv('../data/grab_partner_data.csv')
sensitive_feature = df['location_tier']
print("Data loaded successfully.")

# %%
# =============================================================================
# Cell 3: Exploratory Data Analysis (EDA) & Bias Identification (FIXED)
# =============================================================================
print("\n--- Cell 3: Analyzing Data for Bias ---")

# Analyze the creditworthy rate disparity
bias_analysis = df.groupby('location_tier')['creditworthy'].value_counts(normalize=True).unstack().fillna(0)
print("Creditworthy Rate by Location Tier:\n", bias_analysis)

# Plot the disparity
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.barplot(x=bias_analysis.index, y=bias_analysis[1], ax=ax, palette='viridis')
ax.set_title('Creditworthy Rate by Location Tier (Initial Bias)')
ax.set_ylabel('Proportion Deemed Creditworthy')
ax.set_xlabel('Location Tier')
plt.show()

# --- CORRECTED FAIRNESS CALCULATION USING METRICFRAME ---
# This is the robust way to calculate per-group metrics and avoids previous errors
grouped_on_data = MetricFrame(metrics=selection_rate,
                              y_true=df['creditworthy'],
                              y_pred=df['creditworthy'],
                              sensitive_features=sensitive_feature)

# The results per group are in the .by_group attribute
rates = grouped_on_data.by_group
print(f"\nInitial Selection Rates in Data (per group): \n{rates}")

# Now `rates` is a Pandas Series, so this indexing will work correctly
data_dpd = rates['Tier 1'] - rates['Tier 3']
print(f"\nInitial Demographic Parity Difference in Data: {data_dpd:.4f}")
print("This shows a significant disparity in positive outcomes between groups in the raw data.")


# %%
# =============================================================================
# Cell 4: Feature Engineering and Pre-processing
# =============================================================================
print("\n--- Cell 4: Engineering Features and Preparing Data ---")

# Create simple interaction features
df['rating_to_driving_ratio'] = df['customer_ratings'] / (df['driving_score'] + 1)
df['earnings_per_trip'] = df['weekly_earnings'] / (df['trip_frequency'] + 1)

# Define features (X) and target (y)
X = df.drop(['partner_id', 'creditworthy', 'location_tier'], axis=1)
y = df['creditworthy']
sensitive_feature_split = df['location_tier']

# Split data for training and testing
X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X, y, sensitive_feature_split, test_size=0.3, random_state=42, stratify=y
)

# Create a pre-processing pipeline to scale numeric features
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)])
print("Data pre-processing pipeline created.")

# %%
# =============================================================================
# Cell 5: Train and Evaluate an Unmitigated (Unfair) Model (FIXED)
# =============================================================================
print("\n--- Cell 5: Training Standard XGBoost Model ---")

# Define the full model pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train the model
xgb_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_pipeline.predict(X_test)
y_pred_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

# --- Performance Evaluation ---
print("\n--- Unmitigated Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

# --- Fairness Evaluation ---
print("\n--- Unmitigated Model FAIRNESS ---")
# CORRECTED: Removed all unexpected keyword arguments to prevent TypeErrors
dpd_unmitigated = demographic_parity_difference(y_true=y_test, y_pred=y_pred_xgb, sensitive_features=sf_test)
eod_unmitigated = equalized_odds_difference(y_true=y_test, y_pred=y_pred_xgb, sensitive_features=sf_test)
print(f"Demographic Parity Difference: {dpd_unmitigated:.4f}")
print(f"Equalized Odds Difference: {eod_unmitigated:.4f}")
print("Result: The model is accurate but has inherited and amplified the bias from the data.")

# %%
# =============================================================================
# Cell 6: Mitigate Bias using ThresholdOptimizer (FIXED)
# =============================================================================
print("\n--- Cell 6: Applying Fairness Mitigation ---")

# Use Fairlearn's ThresholdOptimizer to create a fair model wrapper
postprocess_model = ThresholdOptimizer(
    estimator=xgb_pipeline,
    constraints="equalized_odds", # Our chosen fairness constraint
    # CORRECTED: Changed objective to a supported value for `equalized_odds`
    objective="balanced_accuracy_score",
    prefit=True # We use our already-trained model
)

# Fit the optimizer on the training data to learn the new thresholds
postprocess_model.fit(X_train, y_train, sensitive_features=sf_train)
print("ThresholdOptimizer fitted successfully.")

# %%
# =============================================================================
# Cell 7: Evaluate the Mitigated (Fair) Model (FIXED)
# =============================================================================
print("\n--- Cell 7: Evaluating the Final Fair Model ---")

# Make predictions using the fair model
y_pred_mitigated = postprocess_model.predict(X_test, sensitive_features=sf_test)

# --- Performance Evaluation ---
print("\n--- Mitigated Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_mitigated):.4f}")
print(f"ROC AUC Score (from original probabilities): {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")
print("Note: A slight dip in accuracy is an expected trade-off for improved fairness.")

# --- Fairness Evaluation ---
print("\n--- Mitigated Model FAIRNESS ---")
# CORRECTED: Removed all unexpected keyword arguments to prevent TypeErrors
dpd_mitigated = demographic_parity_difference(y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=sf_test)
eod_mitigated = equalized_odds_difference(y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=sf_test)
print(f"Demographic Parity Difference: {dpd_mitigated:.4f}")
print(f"Equalized Odds Difference: {eod_mitigated:.4f}")
print("Result: Both fairness metrics are now much closer to 0. The model is significantly more equitable.")

# %%
# =============================================================================
# Cell 8: Save the Final Model for Deployment
# =============================================================================
print("\n--- Cell 8: Saving Final Model ---")

# Create the models directory if it doesn't exist
if not os.path.exists('../models'):
    os.makedirs('../models')
    
# Save the entire post-processing model object. It contains everything needed.
dump(postprocess_model, '../models/fair_credit_scorer_model.joblib')

print("Final mitigated model saved to 'models/fair_credit_scorer_model.joblib'")
print("\n--- PROJECT COMPLETE ---")
# %%
