import pandas as pd
import numpy as np
import os

def generate_realistic_data(num_partners=5000, output_path='./data/grab_partner_data.csv'):
    """
    Generates a realistic, simulated dataset for the credit scoring model.

    This function creates a dataset where bias is present but not perfectly
    deterministic. It simulates features with realistic overlap and noise,
    and the target variable ('creditworthy') is determined by performance metrics
    that are themselves implicitly correlated with the sensitive attribute.
    This creates a solvable, real-world fairness problem.
    """
    print("--- Generating New, More Realistic Dataset ---")
    np.random.seed(42)

    if not os.path.exists('./data'):
        os.makedirs('./data')

    location_tiers = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], num_partners, p=[0.2, 0.5, 0.3])

    # Generate features with different means per tier but significant variance to create overlap.
    earnings_mean = np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [450, 400, 380])
    weekly_earnings = np.random.normal(earnings_mean, 150, num_partners).round(2).clip(50)

    trip_frequency = np.random.poisson(np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [60, 50, 45]), num_partners)
    customer_ratings = np.random.normal(np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [4.8, 4.7, 4.6]), 0.25).clip(3.5, 5.0).round(2)
    driving_score = np.random.normal(np.select([location_tiers == 'Tier 1', location_tiers == 'Tier 2', location_tiers == 'Tier 3'], [15, 20, 25]), 10).clip(0, 100).round()

    # The target variable's probability is based only on performance.
    # The bias emerges implicitly from the correlation between features and the sensitive attribute.
    base_probability = 0.6
    performance_score = (weekly_earnings / 800) + (customer_ratings / 10) - (driving_score / 100)
    final_probability = (base_probability + performance_score).clip(0.05, 0.95)
    creditworthy = (np.random.rand(num_partners) < final_probability).astype(int)

    # Create and Save DataFrame
    df = pd.DataFrame({
        'partner_id': range(1, num_partners + 1), 'weekly_earnings': weekly_earnings, 'trip_frequency': trip_frequency,
        'customer_ratings': customer_ratings, 'driving_score': driving_score, 'location_tier': location_tiers, 'creditworthy': creditworthy
    })
    
    df.to_csv(output_path, index=False)
    print(f"New dataset saved to {output_path}")

if __name__ == '__main__':
    generate_realistic_data()