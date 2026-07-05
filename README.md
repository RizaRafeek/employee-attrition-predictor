# Employee Attrition Predictor

## What it does
A machine learning web app that predicts whether an employee is at risk of leaving a company based on their profile details. Designed for HR Managers to identify at-risk employees early and take proactive retention measures.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit
- Joblib

## How to Run Locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run feature engineering to generate processed data: `python scripts/03_feature_engineering.py`
4. Launch the app: `streamlit run app/app.py`

## Live Demo
Coming soon

## Dataset
IBM HR Analytics Employee Attrition Dataset (Kaggle)

## Models Trained
| Model | F1 | AUC |
|---|---|---|
| Logistic Regression | 0.537 | 0.798 |
| Decision Tree | 0.364 | 0.640 |
| Random Forest | 0.140 | 0.757 |
| XGBoost | 0.456 | 0.757 |

Best model: Logistic Regression