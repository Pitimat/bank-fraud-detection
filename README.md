# Bank Fraud Detection System

## Project Overview
This project focuses on detecting fraudulent credit card transactions
using machine learning techniques.

The main objective is to identify potentially fraudulent transactions
while balancing precision and recall.

## Dataset
Credit Card Fraud Detection Dataset from Kaggle.

The dataset contains anonymized transaction features (V1-V28),
Time, Amount, and the target variable Class.

## Methodology
- Data Cleaning and Exploratory Data Analysis
- Handling Class Imbalance
- Logistic Regression
- Random Forest
- XGBoost
- Time-based Train/Validation/Test Split
- Threshold Optimization
- Model Evaluation
- SHAP-based Model Interpretation
- Probability Calibration

## Best Model
XGBoost Classifier

## Evaluation
The final model was evaluated using a time-based split.

Selected decision threshold: 0.81

Test results:
- Precision: 0.875
- Recall: 0.757
- F1-score: 0.812

Confusion Matrix:
- True Negatives: 56664
- False Positives: 8
- False Negatives: 18
- True Positives: 56

## API
A REST API was developed using FastAPI.

Available endpoints:
- GET /
- POST /predict

Interactive API documentation is available through /docs.

## Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost,
SHAP, FastAPI, Joblib.

## Disclaimer
This project is developed for educational and portfolio purposes.
The model has not been validated for real-world banking deployment.