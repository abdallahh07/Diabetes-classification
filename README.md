# Diabetes Classification

## Problem
Predicting whether a patient has diabetes based on medical measurements.

## Dataset
- Source: Pima Indians Diabetes Dataset
- Features: Glucose, BMI, Age, Blood Pressure, Insulin, etc.
- Target: Outcome (0 = No Diabetes, 1 = Diabetes)

## Models Used
- Random Forest
- K-Nearest Neighbors
- Logistic Regression
- XGBoost
- SVC
- LinearSVC

## Results
| Model | Accuracy |
|---|---|
| Random Forest | 0.753 |
| KNN | 0.701 |
| Logistic Regression | 0.759 |
| XGBoost | 0.740 |
| SVC | 0.772 |
| LinearSVC | 0.740 |

**Best Model: SVC (Accuracy = 0.772)**

## What I Learned
- Comparing 6 classification models in one pipeline
- Tracking best model automatically
- Hyperparameter tuning with RandomizedSearchCV then GridSearchCV
- Visualizing confusion matrix for best model
- Using PolynomialFeatures with classification problems

## Libraries
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
