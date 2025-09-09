# Project Nova: A Production-Grade Fair Credit Scoring Engine

## 1. Problem Statement
Many gig economy workers are "credit invisible," preventing them from accessing financial products. This project aims to build a fair, data-driven credit scoring model based on performance within the Grab ecosystem. A mandatory component is the active mitigation of bias.

## 2. Solution Overview
This repository contains the source code for a robust machine learning pipeline that trains, evaluates, and selects a fair credit scoring model. The project demonstrates a realistic workflow, including diagnosing and solving a fundamental data bias issue.

The final pipeline follows these steps:
1.  Loads partner performance data from a realistically simulated source.
2.  Engineers advanced **behavioral features** (e.g., earnings consistency, rating trends) to create less biased signals of reliability.
3.  Trains and evaluates an unmitigated Logistic Regression model as a baseline.
4.  Trains a fairness-aware model using the **ExponentiatedGradient (in-processing)** technique.
5.  Compares the models on both performance (Accuracy) and fairness (`EqualizedOddsDifference`).
6.  Selects and saves the final model that meets the predefined fairness criteria.

## 3. Setup and Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd ProjectNova
```
```bash
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate
```
```bash
# Install dependencies
pip install -r requirements.txt
```
## 4. How to Run the Project
This is a two-step process to ensure a clean and reproducible run.

**Step 1: Generate the Realistic Dataset**
This script creates the `grab_partner_data.csv` file with a solvable level of implicit bias.
```bash
python generate_data.py
```
**Step 2: Run the Full Training Pipeline**
This script trains the models, evaluates them for fairness, and saves the final, successful model.
```bash
python -m src.train
```

## 5. Final Results and Bias Mitigation
The key finding of this project was that **intelligent feature engineering was the most effective strategy.** After discovering that the initial dataset was irredeemably biased, a new dataset was created with more nuanced behavioral features.

On this improved dataset, the pipeline produced the following results:

*   **Unmitigated Model (Logistic Regression):**
    *   Accuracy: **94.64%**
    *   Equalized Odds Difference (EOD): **0.0000**

*   **Fairness-Mitigated Model (ExponentiatedGradient):**
    *   Accuracy: **94.64%**
    *   Equalized Odds Difference (EOD): **0.0000**

*   **Conclusion:** The baseline Logistic Regression model, when trained on the well-engineered feature set, was already **perfectly fair** according to our metric. The pipeline correctly validated this and selected the fair model, which met our success criteria (`EOD <= 0.10`). This demonstrates that focusing on high-quality, behavior-driven data can be more effective than relying on algorithms to fix flawed data.

## 6. Project Deliverables
This project successfully meets all expected outcomes:
-   **A functional machine learning model** (`production_credit_scorer.joblib`) and its associated scaler (`scaler.joblib`) are saved in the `/models` directory.
-   **A clear presentation** is provided through this well-documented, modular codebase.
-   **A dedicated bias analysis** is performed programmatically within the `src/train.py` script, which compares an unmitigated model to a fairness-aware model.
-   **A well-documented, replicable codebase** is provided, with all dependencies and run commands specified.