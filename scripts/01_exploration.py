import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"C:\Users\Riza\attrition-project\data\WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()
data.info()
data.isnull().sum()
data['Attrition'].value_counts()

#Findings
#shape = 1470,35
#attrition ratio: 1233:237 (~16%)
#26 numeric(int64) columns and 9 categorical(objects

attr = data['Attrition'].value_counts()
plt.bar(attr.index,attr.values)
plt.title("Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()


pos = data[data['Attrition'] == "Yes"]
neg = data[data['Attrition'] == "No"]
sns.histplot(pos['Age'], alpha = 0.5, label = "emp left")
sns.histplot(neg['Age'], alpha = 0.5, label = "emp stayed")
plt.title("Age based Attrition")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

#employees around 25-30 leaves more
#more people stay at the age of 35-40

pos1 = pos['Department'].value_counts() 
data1 = data['Department'].value_counts() 
rate = pos1 / data1
plt.bar(rate.index , rate.values)
plt.title('Attrition rate by department')
plt.xlabel('Department')
plt.ylabel('Rate')
plt.show()

#more people in sales followed by hr department is attritioning

pos2 = pos['JobRole'].value_counts()
data2 = data['JobRole'].value_counts()
rate2 = pos2 / data2
plt.figure(figsize=(12,6))
plt.bar(rate2.index , rate2.values)
plt.title('Attrition rate by jobrole')
plt.xlabel('JobRole')
plt.ylabel('Rate')
plt.xticks(rotation = 45)
plt.show()

#sales rep attritions the most

sns.boxplot(x = "Attrition", y = "MonthlyIncome", data =data)
plt.show()

#the median salary of the ones who attritioned is way less than the people who did not

pos3 = pos['OverTime'].value_counts()
data3 = data['OverTime'].value_counts() 
rate3 = pos3/data3
plt.bar(rate3.index , rate3.values)
plt.title('Attrition rate by Overtime')
plt.xlabel('Overtime')
plt.ylabel('Attrition Rate')
plt.show()

#10% of employee with no overtime attritions
#30% of employee with overtime attritions
#Overtime can also be a factor