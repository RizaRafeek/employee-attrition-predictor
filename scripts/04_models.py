import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/processed_data.csv')
data.shape
data.dtypes

y = data['Attrition']
x = data.drop('Attrition', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

acc = accuracy_score(y_test,y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)

y_proba = model.predict_proba(x_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("AUC-ROC", auc)

cm = confusion_matrix(y_test, y_pred)
print("cm :", cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt ='d', cmap='Blues', xticklabels=['Predicted Stay','PredictedLeave'],yticklabels=['Actual Stay','Actual Leave'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

results = {}
results['Logistic Regression'] = {
    'accuracy' : acc,
    'f1' : f1,
    'auc' : auc
}
print(results)
