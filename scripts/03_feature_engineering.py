import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Riza\attrition-project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace = True)
le = LabelEncoder()
data['Attrition'] = le.fit_transform(data['Attrition'])
data['Gender'] = le.fit_transform(data['Gender'])
data['OverTime'] = le.fit_transform(data['OverTime'])
data = pd.get_dummies(data, columns=['Department','JobRole','MaritalStatus','EducationField','BusinessTravel'], drop_first = True)

data['Satisfactionscore'] = data[['JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']].sum(axis = 1) / 4
data['Satisfactionscore'].describe()
data['Satisfactionscore'].corr(data['Attrition'])
data['EnvironmentSatisfaction'].corr(data['Attrition'])
data['JobSatisfaction'].corr(data['Attrition'])
data['RelationshipSatisfaction'].corr(data['Attrition'])
data['WorkLifeBalance'].corr(data['Attrition'])

data['IncomeToAge'] = data['MonthlyIncome']/data['Age']
print("IncomeToAge = ",data['IncomeToAge'].corr(data['Attrition']))
print("MonthlyIncome =", data['MonthlyIncome'].corr(data['Attrition']))
print("Age = ",data['Age'].corr(data['Attrition']))

data['TenurePerJob'] = data['YearsAtCompany']/(data['NumCompaniesWorked']+1)
print("TenurePerJob = ", data['TenurePerJob'].corr(data['Attrition']))
print("YearsAtCompany = ",  data['YearsAtCompany'].corr(data['Attrition']))
print("NumCompaniesWorked =", data['NumCompaniesWorked'].corr(data['Attrition']))
#increase in number of companies increases attrition
#increase in years at company decreases attrition

data['OverallExperience'] = data['TotalWorkingYears']/(data['Age'] +1)
data['OverallExperience'].corr(data['Attrition'])
data['TotalWorkingYears'].corr(data['Attrition'])
data['Age'].corr(data['Attrition'])
#here since both the values are in negative pull the combined  feature  was  useful 

#printing the correlation between  all the columns
data.corr()

plt.figure(figsize=(20,16))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')

data.corr()['Attrition'].sort_values()
#find the most correlated and least correlated ccolumns to attrition

#dropping the columns which are closer to zero and the engineered feature which is weak as well from tenureperjob weak determinanant called numCompaniesworked
print(data.shape)
data.drop(['HourlyRate','JobRole_Research Scientist','PerformanceRating','IncomeToAge','NumCompaniesWorked','MonthlyIncome'], axis =1, inplace = True)
print(data.shape)

#engineered 4 different featres and found their correlation as well as their individual value's correlation to attrition 
#found the correlation of all the ccolumns to eachother and then constructed a heatmap to visually represent it 
#sorted the columns with respect to their correlation and dropped the columns which had values close to zero -dropped 6 unwanted columns in total including some other irrelevant columns.