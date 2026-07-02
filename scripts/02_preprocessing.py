import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
data = pd.read_csv(r"C:\Users\Riza\attrition-project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.nunique()
data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace = True)
le = LabelEncoder()
data['Attrition'] = le.fit_transform(data['Attrition'])
data['Gender'] = le.fit_transform(data['Gender'])
data['OverTime'] = le.fit_transform(data['OverTime'])
data.dtypes
data = pd.get_dummies(data, columns=['Department','JobRole','MaritalStatus','EducationField','BusinessTravel'], drop_first = True)
data.dtypes
data.shape

y = data['Attrition']
x = data.drop('Attrition', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =42)

x_train.shape
x_test.shape

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[:2])

joblib.dump(scaler, 'models/scaler.pkl')

# 1. The raw dataset was cleaned, encoded, split, and scaled into train/test sets ready for modeling.
# 2. Fit was performed only on training data, not test data, to avoid data leakage where test-set statistics would influence the scaling and inflate model performance.
# 3. The columns are preprocessed and their range are approximately in a similar range.
