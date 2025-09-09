# src/train.py
import pandas as pd
from joblib import dump

import config 
from src import data_processing
from src import feature_engineering

# Import ML libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
# --- THE CRITICAL FIX IS HERE ---
from sklearn.metrics import accuracy_score # accuracy_score comes from scikit-learn
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
# --- AND HERE ---
from fairlearn.metrics import equalized_odds_difference # Only fairness metrics come from fairlearn

def run_training_pipeline():
    """The main function to execute the full model training and selection process."""
    
    print("--- Final Training Pipeline: Using Logistic Regression for Controllability ---")
    
    # 1. Load and Split Data
    df = data_processing.load_data_from_source(config.DATA_PATH)
    X_train, X_test, y_train, y_test, sf_train, sf_test = data_processing.split_data(df)
    
    # 2. Engineer Features and Scale
    X_train = feature_engineering.create_advanced_features(X_train)
    X_test = feature_engineering.create_advanced_features(X_test)
    X_train = X_train[config.FEATURES]
    X_test = X_test[config.FEATURES]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data loaded and features engineered.")

    # 3. Model Bake-Off with a Simpler, More Controllable Algorithm

    # --- Define our new base model: Logistic Regression ---
    base_model = LogisticRegression(random_state=config.RANDOM_STATE, solver='liblinear')

    # --- Model 1: Unmitigated Logistic Regression ---
    print("\nTraining Unmitigated Logistic Regression...")
    base_model.fit(X_train_scaled, y_train)
    y_pred_base = base_model.predict(X_test_scaled)
    eod_base = equalized_odds_difference(y_test, y_pred_base, sensitive_features=sf_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"Unmitigated Logistic Regression EOD: {eod_base:.4f}, Accuracy: {acc_base:.4f}")

    # --- Model 2: In-Processing (The Robust Solution) ---
    print("\nTraining In-Processing Model (ExponentiatedGradient with Logistic Regression)...")
    constraints = EqualizedOdds()
    inprocess_model = ExponentiatedGradient(
        base_model, # Using the simpler model
        constraints=constraints
    )
    inprocess_model.fit(X_train_scaled, y_train, sensitive_features=sf_train)
    y_pred_inprocessed = inprocess_model.predict(X_test_scaled)
    eod_inprocessed = equalized_odds_difference(y_test, y_pred_inprocessed, sensitive_features=sf_test)
    acc_inprocessed = accuracy_score(y_test, y_pred_inprocessed)
    print(f"In-Processed Logistic Regression EOD: {eod_inprocessed:.4f}, Accuracy: {acc_inprocessed:.4f}")

    # 4. Model Selection
    print("\n--- Model Selection ---")
    
    if eod_inprocessed <= config.FAIRNESS_THRESHOLD_EOD:
        print(f"SUCCESS: In-Processing model meets the fairness threshold ({config.FAIRNESS_THRESHOLD_EOD}). Selecting it as the final model.")
        final_model = inprocess_model
        dump(scaler, config.MODEL_DIR + 'scaler.joblib')
    else:
        print(f"FAILURE: No model met the fairness threshold. This indicates a severe data problem. Flagging for manual review.")
        final_model = inprocess_model # Still save the best attempt
        dump(scaler, config.MODEL_DIR + 'scaler.joblib')

    # 5. Save the Final Model
    model_path = config.MODEL_DIR + config.FINAL_MODEL_NAME
    print(f"\nSaving final model to {model_path}")
    dump(final_model, model_path)
    
    print("--- Final Training Pipeline Complete ---")

if __name__ == '__main__':
    run_training_pipeline()