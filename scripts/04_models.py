import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import joblib

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
#print("Accuracy:", acc_dt)
#print(classification_report(y_test, y_pred_dt))
f1_dt = f1_score(y_test, y_pred_dt)
#print('F1 Score:', f1_dt)

y_proba_dt = dt_model.predict_proba(x_test_scaled)[:, 1]
auc_dt = roc_auc_score(y_test, y_proba_dt)
#print("AUC-ROC", auc_dt)

cm_dt = confusion_matrix(y_test, y_pred_dt)
#print("cm :", cm_dt)


#tree keeps on splitting based on the yes/no values of each column and learn how a specific employee left and what was his column values
#instead o learning the broder pattern

plt.figure(figsize=(20,10))
#plot_tree(dt_model, max_depth=3, feature_names=x.columns, class_names=['Stayed','Left'], filled = True)

#Overtime was the root node, which is the single strongest predictor

dt3 = DecisionTreeClassifier(max_depth= 3, random_state = 42)
dt3.fit(x_train_scaled, y_train)
train_acc3 = accuracy_score(y_train, dt3.predict(x_train_scaled))     #Accuracy on trained data
test_acc3 = accuracy_score(y_test, dt3.predict(x_test_scaled))        #accuracy on test data
#print(train_acc3)
#print(test_acc3)
f1_dt3 = f1_score(y_test, dt3.predict(x_test_scaled))
#print("F1 (depth=3):", f1_dt3)
#print(confusion_matrix(y_test, dt3.predict(x_test_scaled)))

dt5 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt5.fit(x_train_scaled, y_train)
train_acc5 = accuracy_score(y_train, dt5.predict(x_train_scaled))
test_acc5 = accuracy_score(y_test, dt5.predict(x_test_scaled))
#print("train_acc5:" ,train_acc5)
#print("test_acc5 :", test_acc5)
f1_dt5 = f1_score(y_test, dt5.predict(x_test_scaled))
#print("F1 depth=5:", f1_dt5)

dt10 = DecisionTreeClassifier(max_depth=10, random_state=42)
dt10.fit(x_train_scaled, y_train)
train_acc10 = accuracy_score(y_train, dt10.predict(x_train_scaled))
test_acc10 = accuracy_score(y_test, dt10.predict(x_test_scaled))
#print("train_acc10 :", train_acc10)
#print("test_acc10 :", test_acc10)
f1_dt10 = f1_score(y_test, dt10.predict(x_test_scaled))
#print("F1 depth=10:", f1_dt10)

dt_none = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_none.fit(x_train_scaled, y_train)
train_acc_none = accuracy_score(y_train, dt_none.predict(x_train_scaled))
test_acc_none = accuracy_score(y_test, dt_none.predict(x_test_scaled))
#print("train_acc_none: ",train_acc_none)
#print("test_acc_none :", test_acc_none)
f1_dt_none = f1_score(y_test, dt_none.predict(x_test_scaled))
#print("F1 depth=None:", f1_dt_none)

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
#plt.show()

# A Decision Tree with unlimited depth keeps splitting until it perfectly memorizes the training data - at max_depth=None, train accuracy hit 1.0 (100%), but test accuracy was only 0.79.
# This gap between near-perfect training performance and weaker test performance is overfitting: the tree learned overly specific rules tied to individual training examples instead of general patterns.
# Limiting depth actually helped here - max_depth=3 had the lowest train accuracy (0.85) but the HIGHEST test accuracy (0.854), better than every deeper tree, because a simpler tree generalizes better to unseen employees.

results['Decision Tree (best depth=10)'] = {
    'accuracy': test_acc10,
    'f1': f1_dt10,
    'auc': roc_auc_score(y_test, dt10.predict_proba(x_test_scaled)[:, 1])
}
#print(results)

#on imbalanced dataset, prioritise f1(how many actual leavers were caught(recall)against precision) over accuracy
#on nearly balanced dataset, maybe you can consider accuracy

rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_model.fit(x_train_scaled, y_train)
y_pred_rf = rf_model.predict(x_test_scaled)

acc_rf = accuracy_score(y_test, y_pred_rf)
#print("Accuracy Score(rf): ", acc_rf)
f1_rf = f1_score(y_test, y_pred_rf)
#print("F1 Score(rf): ",f1_rf)
y_proba_rf = rf_model.predict_proba(x_test_scaled)[:,1]
auc_rf = roc_auc_score(y_test, y_proba_rf)
#print("Auc(rf): ",auc_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)
#print("Confusion Matric(rf):", cm_rf)
#print("classification report(rf):", classification_report(y_test,y_pred_rf))

results['Random Forest'] = {
    'accuracy': acc_rf,
    'f1': f1_rf,
    'auc': auc_rf
}
#print(results)
#results before we balance the random forest
#right at this stage the random forest performed worse than the previous models

rf_model_balanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model_balanced.fit(x_train_scaled, y_train)
y_pred_rf_bal = rf_model_balanced.predict(x_test_scaled)

acc_rf_bal = accuracy_score(y_test, y_pred_rf_bal)
f1_rf_bal = f1_score(y_test, y_pred_rf_bal)
y_proba_rf_bal = rf_model_balanced.predict_proba(x_test_scaled)[:, 1]
auc_rf_bal = roc_auc_score(y_test, y_proba_rf_bal)
cm_rf_bal = confusion_matrix(y_test, y_pred_rf_bal)

#print("Accuracy (rf balanced):", acc_rf_bal)
#print("F1 (rf balanced):", f1_rf_bal)
#print("AUC (rf balanced):", auc_rf_bal)
#print("Confusion Matrix (rf balanced):", cm_rf_bal)

# Tried class_weight='balanced' as a quick fix - improved F1 slightly (0.140 -> 0.186) and AUC (0.757 -> 0.784), but still well below Logistic Regression's F1 of 0.537.
# This suggests the imbalance problem needs a stronger fix than just class weighting alone (e.g. SMOTE oversampling, or adjusting the classification threshold) - worth revisiting later in the project.

importances = rf_model.feature_importances_
#print(importances)
feat_importance = pd.Series(importances, index=x.columns) #for an indexed list
feat_importance_sorted = feat_importance.sort_values(ascending=False)
print(feat_importance_sorted)

# DailyRate and MonthlyRate rank moderately high in feature importance, but MonthlyIncome itself was dropped in Day 3 based on IncomeToAge's weak correlation.
# Since IncomeToAge was later also dropped (weaker than its components), the salary signal may have been lost entirely rather than replaced - worth reconsidering MonthlyIncome in a future iteration.

top15 = feat_importance_sorted.head(15)
plt.figure(figsize=(10,8))
plt.barh(top15.index, top15.values)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importances - Rndom Forest')
plt.gca().invert_yaxis()        #invert :the most ranked to be at the top
#plt.show()

results_df = pd.DataFrame(results).T
print(results_df)
#Sometimes a simple model can be more apt for a problem than sophisticated ones.It all depends on the problem we are soling and the datset we are handling

xgb = XGBClassifier(n_estimators=100, learning_rate = 0.1, max_depth =4, random_state =42)
xgb.fit(x_train_scaled, y_train)
y_pred_xgb = xgb.predict(x_test_scaled)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
y_proba_xgb = xgb.predict_proba(x_test_scaled)[:,1]
auc_xgb = roc_auc_score(y_test, y_proba_xgb)
xgb_conf = confusion_matrix(y_test, y_pred_xgb)
print("acc_xgb: ",acc_xgb)
print("f1_xgb: ",f1_xgb)
print("auc_xgb: ", auc_xgb)
print("auc_conf: ",xgb_conf)

xgb2 = XGBClassifier(n_estimators=100, learning_rate = 0.3, max_depth =4, random_state =42)
xgb2.fit(x_train_scaled, y_train)
y_pred_xgb2 = xgb2.predict(x_test_scaled)

acc_xgb2 = accuracy_score(y_test, y_pred_xgb2)
f1_xgb2 = f1_score(y_test, y_pred_xgb2)
y_proba_xgb2 = xgb2.predict_proba(x_test_scaled)[:,1]
auc_xgb2 = roc_auc_score(y_test, y_proba_xgb2)
xgb_conf2 = confusion_matrix(y_test, y_pred_xgb2)
print("acc_xgb2: ",acc_xgb2)
print("f1_xgb2: ",f1_xgb2)
print("auc_xgb2: ", auc_xgb2)
print("xgb2_conf: ",xgb_conf2)

xgb3 = XGBClassifier(n_estimators=100, learning_rate = 0.01, max_depth =4, random_state =42)
xgb3.fit(x_train_scaled, y_train)
y_pred_xgb3 = xgb3.predict(x_test_scaled)

acc_xgb3 = accuracy_score(y_test, y_pred_xgb3)
f1_xgb3 = f1_score(y_test, y_pred_xgb3)
y_proba_xgb3 = xgb3.predict_proba(x_test_scaled)[:,1]
auc_xgb3 = roc_auc_score(y_test, y_proba_xgb3)
xgb_conf3 = confusion_matrix(y_test, y_pred_xgb3)
print("acc_xgb3: ",acc_xgb3)
print("f1_xgb3: ",f1_xgb3)
print("auc_xgb3: ", auc_xgb3)
print("xgb3_conf: ",xgb_conf3)

#the learning rate 0.01 performs the worst among three.
#because the learning rate is very small ,the tree has not learned anything even after 100 trees,it might require 1000s of trees
results['XGBoost'] = {
    'accuracy': acc_xgb,
    'f1': f1_xgb,
    'auc': auc_xgb
}
print(results)

#Logistic Regression performed better in terms of f1 and auc when compared to other models
#Logistic regression adds a coefficient to each features before it derives the possibility.this gives hr an idea on what is contributing to risk of losing an employee

joblib.dump(model, 'models/logistic_regression.pkl')
joblib.dump(dt_model, 'models/decision_tree.pkl')
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(xgb, 'models/xgboost.pkl')