import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('data/processed_data.csv')

y = data['Attrition']
x = data.drop('Attrition', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = joblib.load('models/logistic_regression.pkl')
dt_model = joblib.load('models/decision_tree.pkl')
rf_model = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_test_scaled = scaler.transform(x_test)

y_pred_model = model.predict(x_test_scaled)
y_proba_model = model.predict_proba(x_test_scaled)[:,1]
acc = accuracy_score(y_test, y_pred_model)
f1 = f1_score(y_test, y_pred_model)
auc = roc_auc_score(y_test, y_proba_model)
fpr, tpr, th = roc_curve(y_test, y_proba_model)

y_pred_dt = dt_model.predict(x_test_scaled)
y_proba_dt = dt_model.predict_proba(x_test_scaled)[:,1]
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_proba_dt)
fpr_dt, tpr_dt, th_dt = roc_curve(y_test, y_proba_dt)

y_pred_rf = rf_model.predict(x_test_scaled)
y_proba_rf = rf_model.predict_proba(x_test_scaled)[:,1]
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)
fpr_rf, tpr_rf, th_rf = roc_curve(y_test, y_proba_rf)

y_pred_xgb = xgb.predict(x_test_scaled)
y_proba_xgb = xgb.predict_proba(x_test_scaled)[:,1]
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)
fpr_xgb, tpr_xgb, th_xgb = roc_curve(y_test, y_proba_xgb)

results = {}
results['Logistic Regression'] = {
    'accuracy' : acc,
    'f1' : f1,
    'auc' : auc
}
results['Decision Tree'] = {
    'accuracy': acc_dt,
    'f1': f1_dt,
    'auc': auc_dt
}
results['Random Forest'] = {
    'accuracy': acc_rf,
    'f1': f1_rf,
    'auc': auc_rf
}
results['XGBoost'] = {
    'accuracy': acc_xgb,
    'f1': f1_xgb,
    'auc': auc_xgb
}
results_df = pd.DataFrame(results).T
print(results_df)

plt.figure(figsize=(10,8))
plt.plot(fpr, tpr, label = f"Logistic Regression(AUC = {auc:.3f})")
plt.plot(fpr_dt, tpr_dt, label = f"Decision Tree(AUC ={auc_dt:.3f})")
plt.plot(fpr_rf, tpr_rf, label = f"Random Forest(AUC = {auc_rf:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label = f"XGBoost(AUC = {auc_xgb:.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Random Guess')
plt.title("ROC CURVE")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

#1.Logistic Regression .For this dataset and the problem we are solving, i think the probaablistic 0 and 1s were more performing than the other models
#2.Logistic RegressionIt also adds a coefficient of weights to each feature predicting the targer, so the hr can see which factors are affecting the employee attrition more and work on that
#3.False Negative, Losing an employee is  more loss to the company than giving little incentives to an employee which might stay

joblib.dump(model,"models/best_model.pkl")