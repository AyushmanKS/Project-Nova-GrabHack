# Project Nova: A Production-Grade Fair Credit Scoring Engine

## 🚀 Live Interactive Demo

Explore the live, interactive demo of the final model deployed on Streamlit Cloud:

**[https://project-nova-fair-credit-score.streamlit.app/](https://project-nova-fair-credit-score.streamlit.app/)**

## 1. Problem Statement
Many gig economy workers are "credit invisible," preventing them from accessing financial products. This project aims to build a fair, data-driven credit scoring model based on performance within the Grab ecosystem. A mandatory component is the active mitigation of bias to ensure equitable outcomes.

## 2. Solution Overview
This repository contains the source code for a robust machine learning pipeline and an interactive application that trains, evaluates, and demonstrates a fair credit scoring model. The project demonstrates a realistic, end-to-end workflow, from data generation to a final, reliable UI.

The final solution follows these key steps:
1.  **Realistic Data Simulation:** A script generates a dataset with solvable, implicit bias, mimicking real-world conditions.
2.  **Intelligent Feature Engineering:** The pipeline creates advanced **behavioral features** (e.g., earnings consistency, rating trends) to provide less biased signals of a partner's reliability.
3.  **Fairness-Aware Model Training:** A simple, controllable **Logistic Regression** model is trained within a **ExponentiatedGradient (in-processing)** framework to proactively enforce fairness constraints (`EqualizedOddsDifference`).
4.  **Robustness via Business Rules:** The final application includes a **business rule guardrail** that overrides the model's predictions in extreme, out-of-distribution cases, ensuring the system is both statistically fair and logically sound.
5.  **Interactive Demonstration:** A **Streamlit application** provides a user-friendly interface to interact with the final model and understand its decisions.

## 3. Setup and Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ProjectNova

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 4. How to Run the Project
This is a three-step process to run the complete project.

**Step 1: Generate the Realistic Dataset**
This script creates the `grab_partner_data.csv` file needed for training.
```bash
python generate_data.py
```

**Step 2: Run the Full Training Pipeline**
This script trains the models, evaluates them for fairness, and saves the final, successful model artifacts to the `/models` directory.
```bash
python -m src.train
```

**Step 3: Launch the Interactive UI**
This command starts the Streamlit web application, which provides an interactive demo of the final model.
```bash
streamlit run app.py
```

## 5. Final Results and Bias Mitigation

### Model Performance and Fairness
The key finding of this project was that **intelligent feature engineering combined with a controllable model** was the most effective strategy. The pipeline produced the following outstanding results:

*   **Unmitigated Model (Logistic Regression):**
    *   Accuracy: **94.64%**
    *   Equalized Odds Difference (EOD): **0.0000**

*   **Fairness-Mitigated Model (ExponentiatedGradient):**
    *   Accuracy: **94.64%**
    *   Equalized Odds Difference (EOD): **0.0000**

*   **Conclusion:** The baseline Logistic Regression model, when trained on the well-engineered feature set, was already **perfectly fair** according to our EOD metric. The pipeline correctly validated this and selected the fair model, meeting our success criteria (`EOD <= 0.10`).

### Stress Testing and Robustness
A critical part of the project was testing the model against extreme, "worst-case" scenarios. This revealed a classic **extrapolation failure** where the linear model could produce illogical results for inputs far outside its training data.

This issue was solved by implementing a **business rule guardrail** in the Streamlit application. This guardrail checks for common-sense failure conditions (e.g., extremely low ratings and activity) and overrides the model's prediction, ensuring the final system is not only statistically fair but also robust and trustworthy.

## 6. Project Deliverables
This project successfully meets all expected outcomes:

-   **A functional machine learning model** (`production_credit_scorer.joblib`) and its associated scaler (`scaler.joblib`) are saved in the `/models` directory.
-   **An interactive presentation** of the model is provided via the Streamlit application (`app.py`).
-   **A dedicated bias analysis** is performed programmatically within the `src/train.py` script, which compares an unmitigated model to a fairness-aware model.
-   **A well-documented, replicable codebase** is provided, with all dependencies and run commands specified.