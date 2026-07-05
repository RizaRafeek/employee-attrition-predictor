# PLAN:
# Inputs: HR manager enters raw employee details via sliders and dropdowns
# (Age, MonthlyIncome, Department, JobRole, OverTime, satisfaction scores etc - ~20 fields)
# Output: Attrition probability % + Low/Medium/High risk label, color coded
# Feature engineering: app will compute SatisfactionScore, TenurePerJob, OverallExperience
# from raw inputs, then one-hot encode categoricals to match training columns before predicting

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
st.title("Employee Attrition")

Age = st.slider('Age', 18, 70, 18)
Income = st.slider('Monthly Income', 1000, 20000, 5000)
Years = st.slider('Years At Company', 0, 40, 5)
Distance = st.slider('Distance From Home', 1, 30, 5)
TotalYears = st.slider('Total Working Years', 0, 40, 10)
CurrRoleYears = st.slider('Years in Current Role', 0, 20, 3)
PromoYears = st.slider('Years Since Last Promotion', 0, 15, 1)
CurrMgrYears = st.slider('Years with Current Manager', 0, 20, 3)
NumCompanies = st.slider('Number of Companies Worked',0, 10, 2)
JobSatisfy = st.slider('JobSatisfaction', 1, 4, 3)
EnvSatisfy = st.slider('Environment Satisfaction', 1, 4, 3)
RelaSatisfy = st.slider('Relationship Satisfaction', 1, 4, 3)
Balance = st.slider('Work Life Balance', 1, 4, 3)
StockOptLevel = st.slider('Stock Option Level', 0, 3, 0)
JobLevel = st.slider('Job Level',1, 5, 2)
JobInvolvement = st.slider('Job Involvement', 1, 4, 3)
Education = st.slider('Education', 1, 5, 3)

Gender = st.selectbox('Gender', ['Male', 'Female'])
OverTime = st.selectbox('OverTime', ['Yes', 'No'])
Dept = st.selectbox('Department',['Sales', 'Research & Development', 'Human Resources'])
JobRole = st.selectbox('Job Role', ['Sales Representative','Sales Executive', 'Research Scientist', 'Research Director', 'Manager', 'Manufacturing Director', 'Laboratory Technician', 'Human Resources', 'Healthcare Rep'])
MaritalStatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
EducationField = st.selectbox('Education Field',['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
BusinessTravel = st.selectbox('Business Travel', ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])

if st.button('Predict'):
    input_dict = {
        'Age' : Age,
        'YearsAtCompany' : Years,
        'DistanceFromHome' : Distance,
        'TotalWorkingYears' : TotalYears,
        'YearsInCurrentRole' : CurrRoleYears,
        'YearsSinceLastPromotion' : PromoYears,
        'YearsWithCurrManager' : CurrMgrYears,
        'JobSatisfaction' : JobSatisfy,
        'EnvironmentSatisfaction' : EnvSatisfy ,
        'RelationshipSatisfaction' : RelaSatisfy,
        'WorkLifeBalance' : Balance,
        'StockOptionLevel' : StockOptLevel,
        'JobLevel' : JobLevel ,
        'JobInvolvement' : JobInvolvement,
        'Gender' : Gender,
        'OverTime' : OverTime,
        'Education': Education
    }
    SatisfactionScore = (EnvSatisfy + JobSatisfy + RelaSatisfy + Balance )/4
    TenurePerJob = Years / (NumCompanies + 1)
    OverallExperience = TotalYears / (Age + 1)
    input_dict['SatisfactionScore'] = SatisfactionScore
    input_dict['TenurePerJob'] = TenurePerJob
    input_dict['OverallExperience'] = OverallExperience

    gender_map = {'Female' : 0, 'Male' : 1}
    overtime_map = {'No' : 0, 'Yes' : 1}
    input_dict['Gender'] = gender_map[Gender]
    input_dict['OverTime'] = overtime_map[OverTime]

    input_dict['Department_Research & Development'] = 1 if Dept == 'Research & Development' else 0 
    input_dict['Department_Sales'] = 1 if Dept == 'Sales' else 0

    input_dict['JobRole_Human Resources'] = 1 if JobRole == 'Human Resources' else 0 
    input_dict['JobRole_Laboratory Technician'] = 1 if JobRole == 'Laboratory Technician' else 0
    input_dict['JobRole_Manager'] = 1 if JobRole == 'Manager' else 0
    input_dict['JobRole_Manufacturing Director'] = 1 if JobRole == 'Manufacturing Director' else 0
    input_dict['JobRole_Research Director'] = 1 if JobRole == 'Research Director' else 0
    input_dict['JobRole_Sales Executive'] = 1 if JobRole == 'Sales Executive' else 0
    input_dict['JobRole_Sales Representative'] = 1 if JobRole == 'Sales Representative' else 0
    
    input_dict['MaritalStatus_Married'] = 1 if MaritalStatus == 'Married' else 0
    input_dict['MaritalStatus_Single'] = 1 if MaritalStatus == 'Single' else 0

    input_dict['EducationField_Life Sciences'] = 1 if EducationField == 'Life Sciences' else 0
    input_dict['EducationField_Marketing'] = 1 if EducationField == 'Marketing' else 0
    input_dict['EducationField_Medical'] = 1 if EducationField == 'Medical' else 0
    input_dict['EducationField_Other'] = 1 if EducationField == 'Other' else 0
    input_dict['EducationField_Technical Degree'] = 1 if EducationField == 'Technical Degree' else 0

    input_dict['BusinessTravel_Travel_Frequently'] = 1 if BusinessTravel == 'Travel_Frequently' else 0
    input_dict['BusinessTravel_Travel_Rarely'] = 1 if BusinessTravel ==  'Travel_Rarely' else 0

    input_df = pd.DataFrame([input_dict])
    #st.write(input_df)

    processed = pd.read_csv('data/processed_data.csv')
    training_columns = processed.drop('Attrition', axis=1).columns

    input_df = input_df.reindex(columns=training_columns, fill_value=0)
    #st.write(input_df)

    X_train_full = processed.drop('Attrition', axis =1)
    scaler = StandardScaler()
    scaler.fit(X_train_full)

    input_scaled = scaler.transform(input_df)

    model = joblib.load('models/best_model.pkl')

    prediction_proba = model.predict_proba(input_scaled)[:, -1]
    prob_percent = prediction_proba[0] * 100

    if prob_percent < 30:
        risk_label = "Low"
    elif prob_percent < 60:
        risk_label = "Medium"
    else:
        risk_label = "High"

    if risk_label == "Low":
        st.success(f"Risk Level: {risk_label} ({prob_percent:.1f}%)")
    elif risk_label == "Medium":
        st.warning(f"Risk Level: {risk_label} ({prob_percent:.1f}%)")
    else:
        st.error(f"Risk Level: {risk_label} ({prob_percent:.1f}%)")
