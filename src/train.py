import pandas as pd
from joblib import dump
import project_config as config 
from src import data_processing
from src import feature_engineering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import equalized_odds_difference

def run_training_pipeline():
    """Executes the full model training, evaluation, and selection pipeline."""
    
    print("--- Starting Final Training Pipeline ---")
    
    # 1. Load and prepare data
    df = data_processing.load_data_from_source(config.DATA_PATH)
    
    X_train, X_test, y_train, y_test, sf_train, sf_test = data_processing.split_data(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    X_train = feature_engineering.create_advanced_features(X_train)
    X_test = feature_engineering.create_advanced_features(X_test)
    X_train = X_train[config.FEATURES]
    X_test = X_test[config.FEATURES]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data loaded and features engineered.")

    # 2. Define and train models in a "bake-off"
    base_model = LogisticRegression(random_state=config.RANDOM_STATE, solver='liblinear')

    # Unmitigated baseline model
    print("\nTraining Unmitigated Logistic Regression...")
    base_model.fit(X_train_scaled, y_train)
    y_pred_base = base_model.predict(X_test_scaled)
    eod_base = equalized_odds_difference(y_test, y_pred_base, sensitive_features=sf_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"Unmitigated Model EOD: {eod_base:.4f}, Accuracy: {acc_base:.4f}")

    # In-processing model for fairness
    print("\nTraining In-Processing Fair Model (ExponentiatedGradient)...")
    constraints = EqualizedOdds()
    inprocess_model = ExponentiatedGradient(base_model, constraints=constraints)
    inprocess_model.fit(X_train_scaled, y_train, sensitive_features=sf_train)
    
    inprocess_model.predict_proba = inprocess_model.estimator.predict_proba
    y_pred_inprocessed = inprocess_model.predict(X_test_scaled)
    eod_inprocessed = equalized_odds_difference(y_test, y_pred_inprocessed, sensitive_features=sf_test)
    acc_inprocessed = accuracy_score(y_test, y_pred_inprocessed)
    print(f"In-Processed Model EOD: {eod_inprocessed:.4f}, Accuracy: {acc_inprocessed:.4f}")

    # 3. Model Selection based on fairness criteria
    print("\n--- Model Selection ---")
    if eod_inprocessed <= config.FAIRNESS_THRESHOLD_EOD:
        print(f"SUCCESS: In-Processing model meets the fairness threshold. Selecting as final model.")
        final_model = inprocess_model
        dump(scaler, config.MODEL_DIR + 'scaler.joblib')
    else:
        print(f"FAILURE: No model met the fairness threshold. Flagging for manual review.")
        final_model = inprocess_model
        dump(scaler, config.MODEL_DIR + 'scaler.joblib')

    # 4. Save the final selected model
    model_path = config.MODEL_DIR + config.FINAL_MODEL_NAME
    print(f"\nSaving final model to {model_path}")
    dump(final_model, model_path)
    
    print("--- Final Training Pipeline Complete ---")

if __name__ == '__main__':
    run_training_pipeline()