import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

cl_report = classification_report(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


y_proba = model.predict_proba(x_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_proba)


cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt ='d', cmap='Blues', xticklabels=['Predicted Stay','PredictedLeave'],yticklabels=['Actual Stay','Actual Leave'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')


results = {}
results['Logistic Regression'] = {
    'accuracy' : acc,
    'f1' : f1,
    'auc' : auc
}


# False negatives are more dangerous than false positives in this context.
# A false positive just means wasted retention effort (time, a small raise, a check-in) on someone who was never actually at risk - a minor, recoverable cost.
# A false negative means a genuinely at-risk employee gets no intervention at all, and when they leave, the company faces real costs: recruiting and training a replacement, lost productivity during the gap, and loss of institutional knowledge.
# So the model should ideally be tuned to catch more actual leavers, even if it means accepting more false alarms.

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train_scaled, y_train)
y_pred_dt = dt_model.predict(x_test_scaled)

acc_dt = accuracy_score(y_test,y_pred_dt)
print("Accuracy:", acc_dt)
print(classification_report(y_test, y_pred_dt))
f1_dt = f1_score(y_test, y_pred_dt)
print('F1 Score:', f1_dt)

y_proba_dt = dt_model.predict_proba(x_test_scaled)[:, 1]
auc_dt = roc_auc_score(y_test, y_proba_dt)
print("AUC-ROC", auc_dt)

cm_dt = confusion_matrix(y_test, y_pred_dt)
print("cm :", cm_dt)


#tree keeps on splitting based on the yes/no values of each column and learn how a specific employee left and what was his column values
#instead o learning the broder pattern

plt.figure(figsize=(20,10))
plot_tree(dt_model, max_depth=3, feature_names=x.columns, class_names=['Stayed','Left'], filled = True)

#Overtime was the root node, which is the single strongest predictor

dt3 = DecisionTreeClassifier(max_depth= 3, random_state = 42)
dt3.fit(x_train_scaled, y_train)
train_acc3 = accuracy_score(y_train, dt3.predict(x_train_scaled))     #Accuracy on trained data
test_acc3 = accuracy_score(y_test, dt3.predict(x_test_scaled))        #accuracy on test data
print(train_acc3)
print(test_acc3)
f1_dt3 = f1_score(y_test, dt3.predict(x_test_scaled))
print("F1 (depth=3):", f1_dt3)
print(confusion_matrix(y_test, dt3.predict(x_test_scaled)))

dt5 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt5.fit(x_train_scaled, y_train)
train_acc5 = accuracy_score(y_train, dt5.predict(x_train_scaled))
test_acc5 = accuracy_score(y_test, dt5.predict(x_test_scaled))
print("train_acc5:" ,train_acc5)
print("test_acc5 :", test_acc5)
f1_dt5 = f1_score(y_test, dt5.predict(x_test_scaled))
print("F1 depth=5:", f1_dt5)

dt10 = DecisionTreeClassifier(max_depth=10, random_state=42)
dt10.fit(x_train_scaled, y_train)
train_acc10 = accuracy_score(y_train, dt10.predict(x_train_scaled))
test_acc10 = accuracy_score(y_test, dt10.predict(x_test_scaled))
print("train_acc10 :", train_acc10)
print("test_acc10 :", test_acc10)
f1_dt10 = f1_score(y_test, dt10.predict(x_test_scaled))
print("F1 depth=10:", f1_dt10)

dt_none = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_none.fit(x_train_scaled, y_train)
train_acc_none = accuracy_score(y_train, dt_none.predict(x_train_scaled))
test_acc_none = accuracy_score(y_test, dt_none.predict(x_test_scaled))
print("train_acc_none: ",train_acc_none)
print("test_acc_none :", test_acc_none)
f1_dt_none = f1_score(y_test, dt_none.predict(x_test_scaled))
print("F1 depth=None:", f1_dt_none)

#this shows how increasing the depth of the tree makes the model learn and overfit and do better on training data while declining accuracy in testing data

depths = [3, 5, 10, 20]
train_accs = [0.852, 0.897, 0.987, 1.000]
test_accs = [0.854, 0.816, 0.806, 0.793]

plt.figure(figsize=(8,6))
plt.plot(depths, train_accs, marker='o', label='Train Accuracy')
plt.plot(depths, test_accs, marker='o', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Train vs Tesst Accuracy by Tree Depth')
plt.legend()
plt.show()

# A Decision Tree with unlimited depth keeps splitting until it perfectly memorizes the training data - at max_depth=None, train accuracy hit 1.0 (100%), but test accuracy was only 0.79.
# This gap between near-perfect training performance and weaker test performance is overfitting: the tree learned overly specific rules tied to individual training examples instead of general patterns.
# Limiting depth actually helped here - max_depth=3 had the lowest train accuracy (0.85) but the HIGHEST test accuracy (0.854), better than every deeper tree, because a simpler tree generalizes better to unseen employees.

results['Decision Tree (best depth=10)'] = {
    'accuracy': test_acc10,
    'f1': f1_dt10,
    'auc': roc_auc_score(y_test, dt10.predict_proba(x_test_scaled)[:, 1])
}
print(results)

#on imbalanced dataset, prioritise f1(how many actual leavers were caught(recall)against precision) over accuracy
#on nearly balanced dataset, maybe you can consider accuracy