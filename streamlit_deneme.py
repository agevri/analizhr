import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import RobustScaler, LabelEncoder

st.set_page_config(layout="wide")


# bi kere çalıştır ön belleğe al tekrar tekrar çalıştırma
@st.cache_data
def get_data():
    df= pd.read_csv("HR_data.csv")
    return df

def get_model():
    model=joblib.load("HR_Attrition_voting_clf2.pkl")
    return model

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

# Aykırı değer limitleri
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# One Hot Encoder
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def hr_data_prep(df):
    #Uppercase the variable names
    df.columns = df.columns.astype("str").str.upper()

    # Eksik değerlerin doldurulması fonksiyonu
    #quick_missing_imp(df, num_method="median", target="ATTRITION")

    # Removing useless variables
    df["ATTRITION"] = df["ATTRITION"].apply(lambda x: 1 if x == "Yes" else 0)
    df["OVERTIME"] = df["OVERTIME"].apply(lambda x: 1 if x == "Yes" else 0)
    df = df.drop(["OVER18", "STANDARDHOURS", "EMPLOYEECOUNT", "EMPLOYEENUMBER"], axis=1)

    df['NEW_AGE_GROUP'] = pd.cut(df['AGE'], bins=[17, 30, 40, 50, 60], labels=['18_30', '30_40', '40_50', '50_60'])
    df["NEW_AGE_GROUP"].value_counts()

    # Eve uzaklık Grupları
    df["NEW_DISTANCE_GROUP"] = pd.cut(df["DISTANCEFROMHOME"], bins=[0, 2, 5, 10, 20, 30],
                                      labels=["YURUME", "KISA", "ORTA", "ORTAUZAK", "UZAK"])
    df["NEW_DISTANCE_GROUP"].value_counts()

    # Maaş Grupları
    df["NEW_SALARY_GROUP"] = pd.cut(df["MONTHLYINCOME"], bins=[0, 2000, 5000, 10000, 20000],
                                    labels=["0_2000", "2000_5000", "5000_10000", "10000_20000"])
    df["NEW_SALARY_GROUP"].value_counts()

    # Çalışan Yılı Grupları
    df['NEW_YEARS_AT_COMPANY_GROUP'] = pd.cut(df['YEARSATCOMPANY'], bins=[-1, 2, 5, 10, 20, float('inf')],
                                              labels=['0_2', '2_5', '5_10', '10_20', 'over20'])
    df['NEW_YEARS_AT_COMPANY_GROUP'].value_counts()

    # Total Satisfaction
    df['NEW_TOTALSATISFACTION_MEAN'] = (df['RELATIONSHIPSATISFACTION'] + df['ENVIRONMENTSATISFACTION'] + df[
        'JOBSATISFACTION'] + df['JOBINVOLVEMENT'] + df['WORKLIFEBALANCE']) / 5

    # İş tatmini ve performans ilişkisini değerlendiren değişken
    df['NEW_SATISFACTION_PERFORMANCE_SCORE'] = df['NEW_TOTALSATISFACTION_MEAN'] * df['PERFORMANCERATING']
    df['NEW_SATISFACTION_PERFORMANCE_SCORE'].value_counts()

    # Eğitim seviyesi ile iş doyumu arasındaki ilişkiyi değerlendiren değişken
    df['NEW_EDUCATION_SATISFACTION_SCORE'] = df['EDUCATION'] * df['NEW_TOTALSATISFACTION_MEAN']
    df['NEW_EDUCATION_SATISFACTION_SCORE'].value_counts()

    # Sadakat
    df["NEW_FIDELITY_SCORE"] = (df["NUMCOMPANIESWORKED"]) / (df["TOTALWORKINGYEARS"]).replace(0, 1)

    # Mevcut rolde geçirilen süre ile şirketteki toplam deneyimi arasındaki ilişkiyi değerlendiren değişken
    df['NEW_ROLE_EXPERIENCE_RATIO_SCORE'] = (df['YEARSINCURRENTROLE']) / (df['YEARSATCOMPANY']).replace(0, 1)

    # Eğitim seviyesi ile iş seviyesi arasındaki ilişkiyi değerlendiren değişken
    df['NEW_EDUCATION_JOBLEVEL'] = df['EDUCATION'] * df['JOBLEVEL']

    # Fazla mesai yapma durumu ile iş-yaşam dengesi arasındaki ilişkiyi değerlendiren değişken
    df['NEW_OVERTIME_LIFE_BALANCE'] = df['OVERTIME'].astype(str) + df['WORKLIFEBALANCE'].astype(str)

    # Yaş ile aylık gelir arasındaki ilişkiyi değerlendiren değişken
    df['NEW_AGE_INCOME'] = df['MONTHLYINCOME'] / df['AGE']

    # Cinsiyet ile iş doyumu arasındaki ilişkiyi değerlendiren değişken
    # df['NEW_GENDER_SATISFACTION'] = df['GENDER'].astype(str) + df['JOBSATISFACTION'].astype(str)

    # Yıllık gelir ile toplam çalışma deneyimi arasındaki ilişkiyi değerlendiren değişken
    df['NEW_INCOME_EXPERIENCE_RATIO_SCORE'] = df['MONTHLYINCOME'] / (df['TOTALWORKINGYEARS'] + 1)

    # Medeni durumuna göre StockOptionLevel ilişkisi (Bunu kategorileri numeriğe dönüştürüp yapabiliriz.Bekar = 0 Boşanmış=1 Evli = 2 gibi)

    # df["NEW_MARITAL_STOCK"] = df["MARITALSTATUS"] * df["STOCKOPTIONLEVEL"]

    # sinem--- Uzun mu oldu emin olamadım bakalım birlikte karar verelim :)

    # medeni durum ve eve olan mesafe

    df.loc[(df["MARITALSTATUS"] == 'Single') & (df["NEW_DISTANCE_GROUP"] == 'YURUME'), "NEW_MARITAL_DISTANCE"] = "Single_YURUME"
    df.loc[(df["MARITALSTATUS"] == 'Single') & (df["NEW_DISTANCE_GROUP"] == 'KISA'), "NEW_MARITAL_DISTANCE"] = "Single_KISA"
    df.loc[(df["MARITALSTATUS"] == 'Single') & (df["NEW_DISTANCE_GROUP"] == 'ORTA'), "NEW_MARITAL_DISTANCE"] = "Single_ORTA"
    df.loc[(df["MARITALSTATUS"] == 'Single') & (df["NEW_DISTANCE_GROUP"] == 'ORTAUZAK'), "NEW_MARITAL_DISTANCE"] = "Single_ORTAUZAK"
    df.loc[(df["MARITALSTATUS"] == 'Single') & (df["NEW_DISTANCE_GROUP"] == 'UZAK'), "NEW_MARITAL_DISTANCE"] = "Single_UZAK"
    df.loc[(df["MARITALSTATUS"] == 'Married') & (df["NEW_DISTANCE_GROUP"] == 'YURUME'), "NEW_MARITAL_DISTANCE"] = "Married_YURUME"
    df.loc[(df["MARITALSTATUS"] == 'Married') & (df["NEW_DISTANCE_GROUP"] == 'KISA'), "NEW_MARITAL_DISTANCE"] = "Married_KISA"
    df.loc[(df["MARITALSTATUS"] == 'Married') & (df["NEW_DISTANCE_GROUP"] == 'ORTA'), "NEW_MARITAL_DISTANCE"] = "Married_ORTA"
    df.loc[(df["MARITALSTATUS"] == 'Married') & (df["NEW_DISTANCE_GROUP"] == 'ORTAUZAK'), "NEW_MARITAL_DISTANCE"] = "Married_ORTAUZAK"
    df.loc[(df["MARITALSTATUS"] == 'Married') & (df["NEW_DISTANCE_GROUP"] == 'UZAK'), "NEW_MARITAL_DISTANCE"] = "Married_UZAK"
    df.loc[(df["MARITALSTATUS"] == 'Divorced') & (df["NEW_DISTANCE_GROUP"] == 'YURUME'), "NEW_MARITAL_DISTANCE"] = "Divorced_YURUME"
    df.loc[(df["MARITALSTATUS"] == 'Divorced') & (df["NEW_DISTANCE_GROUP"] == 'KISA'), "NEW_MARITAL_DISTANCE"] = "Divorced_KISA"
    df.loc[(df["MARITALSTATUS"] == 'Divorced') & (df["NEW_DISTANCE_GROUP"] == 'ORTA'), "NEW_MARITAL_DISTANCE"] = "Divorced_ORTA"
    df.loc[(df["MARITALSTATUS"] == 'Divorced') & (df["NEW_DISTANCE_GROUP"] == 'ORTAUZAK'), "NEW_MARITAL_DISTANCE"] = "Divorced_ORTAUZAK"
    df.loc[(df["MARITALSTATUS"] == 'Divorced') & (df["NEW_DISTANCE_GROUP"] == 'UZAK'), "NEW_MARITAL_DISTANCE"] = "Divorced_UZAK"

    # yaş ve maaş

    df.loc[(df["NEW_AGE_GROUP"] == '18_30') & (df["NEW_SALARY_GROUP"] == '0_2000'), "NEW_AGE_SALARY"] = "18_30_LOWSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '18_30') & (df["NEW_SALARY_GROUP"] == '2000_5000'), "NEW_AGE_SALARY"] = "18_30_MIDDLESALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '18_30') & (df["NEW_SALARY_GROUP"] == '5000_10000'), "NEW_AGE_SALARY"] = "18_30_HIGHSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '18_30') & (df["NEW_SALARY_GROUP"] == '10000_20000'), "NEW_AGE_SALARY"] = "18_30_HIGHPLUSSALARY"

    df.loc[(df["NEW_AGE_GROUP"] == '30_40') & (df["NEW_SALARY_GROUP"] == '0_2000'), "NEW_AGE_SALARY"] = "30_40_LOWSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '30_40') & (df["NEW_SALARY_GROUP"] == '2000_5000'), "NEW_AGE_SALARY"] = "30_40_MIDDLESALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '30_40') & (df["NEW_SALARY_GROUP"] == '5000_10000'), "NEW_AGE_SALARY"] = "30_40_HIGHSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '30_40') & (df["NEW_SALARY_GROUP"] == '10000_20000'), "NEW_AGE_SALARY"] = "30_40_HIGHPLUSSALARY"

    df.loc[(df["NEW_AGE_GROUP"] == '40_50') & (df["NEW_SALARY_GROUP"] == '0_2000'), "NEW_AGE_SALARY"] = "40_50_LOWSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '40_50') & (df["NEW_SALARY_GROUP"] == '2000_5000'), "NEW_AGE_SALARY"] = "40_50_MIDDLESALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '40_50') & (df["NEW_SALARY_GROUP"] == '5000_10000'), "NEW_AGE_SALARY"] = "40_50_HIGHSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '40_50') & (df["NEW_SALARY_GROUP"] == '10000_20000'), "NEW_AGE_SALARY"] = "40_50_HIGHPLUSSALARY"

    df.loc[(df["NEW_AGE_GROUP"] == '50_60') & (df["NEW_SALARY_GROUP"] == '0_2000'), "NEW_AGE_SALARY"] = "50_60_LOWSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '50_60') & (df["NEW_SALARY_GROUP"] == '2000_5000'), "NEW_AGE_SALARY"] = "50_60_MIDDLESALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '50_60') & (df["NEW_SALARY_GROUP"] == '5000_10000'), "NEW_AGE_SALARY"] = "50_60_HIGHSALARY"
    df.loc[(df["NEW_AGE_GROUP"] == '50_60') & (df["NEW_SALARY_GROUP"] == '10000_20000'), "NEW_AGE_SALARY"] = "50_60_HIGHPLUSSALARY"

    # Salary to Hourly Rate Ratio
    df['NEW_SALARYTOHOUR_RATIO'] = df['MONTHLYINCOME'] / (df['HOURLYRATE'].replace(0, 1))

    # Education Level and Performance Rating Interaction
    # df['NEW_EDUCATION_PERFORMANCE_INTERACTION'] = df['EDUCATION'] * df['PERFORMANCERATING']

    # Monthly Rate to Daily Rate Ratio
    df['NEW_MONTHLYRATE_DAILYRATE_RATIO'] = df['MONTHLYRATE'] / df['DAILYRATE']

    # Overtime and Monthly Income Interaction
    df['NEW_OVERTIME_MONTHLYINCOME_INTERACTION'] = df['OVERTIME'].map({'Yes': 1, 'No': 0}) * df['MONTHLYINCOME']

    # Monthly Income to Performance Rating Ratio
    df['NEW_MONTHLYINCOME_PERFORMANCERATING_RATIO'] = df['MONTHLYINCOME'] / df['PERFORMANCERATING']

    # Salary to Distance Ratio
    df['NEW_SALARY_DISTANCE_RATIO'] = df['MONTHLYINCOME'] / (df['DISTANCEFROMHOME'].replace(0, 1))

    # DailyRate and PerformanceRating Interaction
    df['NEW_DAILYRATE_PERFORMANCERATING_INTERACTION'] = df['DAILYRATE'] * df['PERFORMANCERATING']

    # Education Level and Hourly Rate Interaction
    df['NEW_EDUCATION_HOURLYRATE_INTERACTION'] = df['EDUCATION'] * df['HOURLYRATE']

    # Distance from Home and Overtime Interaction
    df['NEW_DISTANCE_OVERTIME_INTERACTION'] = df['DISTANCEFROMHOME'] * df['OVERTIME'].map({'Yes': 1, 'No': 0})

    # Monthly Income to Percent Salary Hike Ratio
    df['NEW_MONTHLYINCOME_SALARYHIKE_RATIO'] = df['MONTHLYINCOME'] / (df['PERCENTSALARYHIKE'].replace(0, 1))

    # Feature 3: Job Satisfaction and Performance Interaction
    # df['NEW_JOBSATISFACTION_PERFORMANCE'] = df['JOBSATISFACTION'] * df['PERFORMANCERATING']

    # Feature 6: Job Level and Years at Company Interaction
    # df['NEW_JOBLEVEL_YEARSATCOMPANY_INTERACTION'] = df['JOBLEVEL'] * df['YEARSATCOMPANY']

    # OverTime and Job Satisfaction Interaction
    df['NEW_OVERTIME_SATISFACTION'] = df['OVERTIME'] * df['JOBSATISFACTION']

    # Age and Years in Current Role Interaction
    df['NEW_AGE_YEARSINCURRENTROLE'] = df['AGE'] * df['YEARSINCURRENTROLE']

    # Gender and Job Satisfaction Interaction
    # df['NEW_GENDER_SATISFACTION'] = df['GENDER'].map({'Male': 1, 'Female': 0}) * df['JOBSATISFACTION']

    # Marital Status and Environment Satisfaction Interaction
    df['NEW_MARITALSTATUS_ENVIRONMENTSATISFACTION'] = df['MARITALSTATUS'].map({'Single': 1, 'Married': 2, 'Divorced': 3}) * df['ENVIRONMENTSATISFACTION']

    # Job Role and Monthly Income Interaction
    job_role_income_interaction = df.groupby('JOBROLE')['MONTHLYINCOME'].transform('mean')
    df['NEW_JOBROLE_INCOME_INTERACTION'] = df['JOBROLE'].map(job_role_income_interaction)

    # Years with Current Manager and Salary Interaction
    df['NEW_YEARSWITHCURRENTMANAGER_SALARY_INTERACTION'] = df['YEARSWITHCURRMANAGER'] * df['MONTHLYINCOME']

    # Job Level and Environment Satisfaction Interaction
    df['NEW_JOBLEVEL_ENVIRONMENTSATISFACTION_INTERACTION'] = df['JOBLEVEL'] * df['ENVIRONMENTSATISFACTION']

    cat_cols= ['BUSINESSTRAVEL', 'DEPARTMENT', 'EDUCATIONFIELD', 'GENDER', 'JOBROLE', 'MARITALSTATUS',
               'NEW_OVERTIME_LIFE_BALANCE', 'NEW_MARITAL_DISTANCE', 'NEW_AGE_SALARY', 'ATTRITION', 'EDUCATION',
               'ENVIRONMENTSATISFACTION', 'JOBINVOLVEMENT', 'JOBLEVEL', 'JOBSATISFACTION', 'OVERTIME', 'PERFORMANCERATING',
               'RELATIONSHIPSATISFACTION', 'STOCKOPTIONLEVEL', 'TRAININGTIMESLASTYEAR', 'WORKLIFEBALANCE', 'NEW_AGE_GROUP',
               'NEW_DISTANCE_GROUP', 'NEW_SALARY_GROUP', 'NEW_YEARS_AT_COMPANY_GROUP', 'NEW_OVERTIME_MONTHLYINCOME_INTERACTION',
               'NEW_DISTANCE_OVERTIME_INTERACTION', 'NEW_OVERTIME_SATISFACTION', 'NEW_MARITALSTATUS_ENVIRONMENTSATISFACTION', 'NEW_JOBROLE_INCOME_INTERACTION']
    num_cols = ['AGE', 'DAILYRATE', 'DISTANCEFROMHOME', 'HOURLYRATE', 'MONTHLYINCOME', 'MONTHLYRATE', 'NUMCOMPANIESWORKED',
                'PERCENTSALARYHIKE', 'TOTALWORKINGYEARS', 'YEARSATCOMPANY', 'YEARSINCURRENTROLE', 'YEARSSINCELASTPROMOTION',
                'YEARSWITHCURRMANAGER', 'NEW_TOTALSATISFACTION_MEAN', 'NEW_SATISFACTION_PERFORMANCE_SCORE', 'NEW_EDUCATION_SATISFACTION_SCORE',
                'NEW_FIDELITY_SCORE', 'NEW_ROLE_EXPERIENCE_RATIO_SCORE', 'NEW_EDUCATION_JOBLEVEL', 'NEW_AGE_INCOME', 'NEW_INCOME_EXPERIENCE_RATIO_SCORE',
                'NEW_SALARYTOHOUR_RATIO', 'NEW_MONTHLYRATE_DAILYRATE_RATIO', 'NEW_MONTHLYINCOME_PERFORMANCERATING_RATIO', 'NEW_SALARY_DISTANCE_RATIO',
                'NEW_DAILYRATE_PERFORMANCERATING_INTERACTION', 'NEW_EDUCATION_HOURLYRATE_INTERACTION', 'NEW_MONTHLYINCOME_SALARYHIKE_RATIO',
                'NEW_AGE_YEARSINCURRENTROLE', 'NEW_YEARSWITHCURRENTMANAGER_SALARY_INTERACTION', 'NEW_JOBLEVEL_ENVIRONMENTSATISFACTION_INTERACTION']

    # Aykırı değerlerin baskılanması
    for col in num_cols:
        if col != "ATTRITION":
            replace_with_thresholds(df, col)

    binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

    for col in binary_cols:
        label_encoder(df, col)

    cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["ATTRITION"]]

    df = one_hot_encoder(df, cat_cols, drop_first=False)

    # log dönüşümü
    for col in num_cols:
        df[col] = np.log1p(df[col])

    # Robust Scaling
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    y = df["ATTRITION"]
    X = df.drop(["ATTRITION"], axis=1)

    return X, y


# her şeyi st üzerinden çağırıyoruz
# st ile yaptığımız her yerde gözükecek
# kolonlardaki gözükmesini istediklerimizi colonlara atayacağız
st.image("Logo-Transparent.png", width=150)

st.header("EMPLOYEE :green[ATTRITION ANALYSIS] ")
# win . ile emoji ekleyebiliyoruz


# TAB HOME
tab_home,tab_dash,tab_vis,tab_model = st.tabs([":green[Home]", ":green[Dashboard]",":green[Graphs]",":green[Model]"])


# sayfayı 2ye bölmek istiyoruz
col_hr, col_dataset=tab_home.columns(2) # kolonlar ana sayfada var ama diğer sekmelerde yok şekilde ayarlandı

col_hr.subheader("PandaSolutions")

# st.text kullanışlı değil
# bunun yerine st.write ya da st.markdown (alt satırlara geçerek yazdırıyor)

col_hr.markdown("Panda Solutions is a consultancy company that provides predictive employee attiriton services on HR analytics. Company x has developed an attrition forecasting model for its employees. The attrition prediction model makes strong predictions about attrition and turnover when the employee's salary, job satisfaction, age, over time etc. are known.")

# markdown da aralara resimle vs atılabiliyor

#col_hr.image("Logo-Transparent.png")

col_dataset.subheader("Sample Dataset")
col_dataset.markdown("This is a fictional dataset created by IBM data scientists.It includes information such as employees' ages, genders, job roles, salaries, education fields and levels, and performance levels. It also includes information such as job involvement, job satisfaction, environmental satisfaction, relationship satisfaction, work life balance obtained from surveys administered to employees.")

dff=get_data()
col_dataset.dataframe(dff)

# TAB vIS
tab_dash.subheader("HR Attrition Dashboard")
tab_dash.image("Attrition.png")
col_graph1, col_graph2=tab_vis.columns(2, gap="small")
# graph1

col_graph1.subheader("Attrition vs Job Role")
fig3 = px.sunburst(dff, path=["Attrition","JobRole"], color="JobRole",color_continuous_scale='RdBu')
col_graph1.plotly_chart(fig3)

#graph2
col_graph1.subheader("Attrition vs Department")
selected_feat=col_graph1.multiselect(label="Select the department", options=dff.Department.unique(), default="Sales")
filtered_df=dff[dff.Department.isin(selected_feat)]

fig= px.bar(data_frame=filtered_df,
            x="Attrition",
            y="Department",
            color="EducationField")

col_graph1.plotly_chart(fig)

# graph3


col_graph2.subheader("Attrition vs Montyly Income")
fig4 = px.violin(dff, x="Attrition", y="MonthlyIncome", color="Attrition")
col_graph2.plotly_chart(fig4)


#graph5
col_graph2.subheader("Attrition vs Gender with over time circumstances")
fig6 = px.bar(dff, x="Gender", y="Attrition",color='OverTime', barmode='group')
col_graph2.plotly_chart(fig6)


# TAB MODEL

model=get_model()

col_select, col_select2=tab_model.columns(2)
age= pd.to_numeric(col_select.number_input("Enter age of employee", step=1,min_value=0))
business_travel = col_select.selectbox("Select the employee's travel frequency", ("Travel_Rarely", "Travel_Frequently", "Non-Travel"))
department = col_select.selectbox("Select the department of employee", ("Human Resources","Sales","Research & Development"))
distancefromhome = col_select.number_input("Enter the employee's commuting distance", step=1,min_value=0)
education = col_select.number_input("Enter the employee's education level in the range of 1-5. (1: 'Below College', 2:'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor')", step=1,min_value=1, max_value=5)
education_field = col_select.selectbox("Select the employee's education field.", ("Human Resources","Life Sciences","Medical", "Marketing","Technical Degree","Other"))
gender = col_select.selectbox("Select the employee's gender", ("Male", "Female"))
jobinvolvement = col_select.number_input("Enter the job innvolvement of employee in the range of 1-4", step=1, min_value=1,max_value=4)
joblevel = col_select.number_input("Enter the job level of employee in the range of 1-5", step=1,min_value=1,max_value=5)
jobrole = col_select.selectbox("Select the employee's job role",("Healthcare Representative","Human Resources", "Laboratory Technician",
                                                                "Manager", "Manufacturing Director", "Research Director",
                                                                "Research Scientist", "Sales Executive", "Sales Representative"))
marital_status=col_select2.selectbox("Select the employee's maritial status", ("Single", "Married", "Divorced"))
monthly_income=col_select2.number_input("Enter the employee's monthly income",step=1,min_value=0)
num_of_comp_worked=col_select2.number_input("Enter the total number of companies the employee has previously worked for", step=1,min_value=0)
over_time=col_select2.selectbox("Select whether the employee works overtime", ("Yes", "No"))
stockoption= col_select2.number_input("Enter the employee's stock option level", step=1,min_value=0)
totalworkingyear=col_select2.number_input("Enter the employee's total years of work", step=1, min_value=0)
trainingtimes=col_select2.number_input("Enter the number of training the employee received in the previous year",step=1,min_value=0)
yearsatcomp=col_select2.number_input("Enter the employee's total years with this company",step=1,min_value=0)
yearscurrentrole=col_select2.number_input("Enter the employee's years of employment in their current job role",step=1,min_value=0)
yearslstpromotion=col_select2.number_input("Enter how many years ago the employee was last promoted",step=1,min_value=0)
yearswithman=col_select2.number_input("Enter the number of years the employee has worked with their current manager.",step=1,min_value=0)


employee = pd.DataFrame({'Age': pd.to_numeric(age), 'Attrition': "NaN", "BusinessTravel": business_travel,
                        "DailyRate": int(dff.DailyRate.mean()), "Department": department, "DistanceFromHome": distancefromhome,
                        "Education": education, "EducationField": education_field, "EmployeeCount": int(dff.EmployeeCount.mean()),
                        "EmployeeNumber": int(dff.EmployeeNumber.mean()), "EnvironmentSatisfaction": int(dff.EnvironmentSatisfaction.mean()), "Gender": gender, "HourlyRate": int(dff.HourlyRate.mean()),
                        "JobInvolvement": jobinvolvement, "JobLevel": joblevel, "JobRole": jobrole, "JobSatisfaction": int(dff.JobSatisfaction.mean()),
                        "MaritalStatus": marital_status, "MonthlyIncome": monthly_income, "MonthlyRate": int(dff.MonthlyRate.mean()), "NumCompaniesWorked": num_of_comp_worked,
                        "Over18": "Y", "OverTime": over_time, "PercentSalaryHike": int(dff.PercentSalaryHike.mean()), "PerformanceRating": int(dff.PerformanceRating.mean()),
                        "RelationshipSatisfaction": int(dff.RelationshipSatisfaction.mean()), "StandardHours": int(dff.StandardHours.mean()), "StockOptionLevel": stockoption, "TotalWorkingYears": totalworkingyear,
                        "TrainingTimesLastYear": trainingtimes, "WorkLifeBalance": int(dff.WorkLifeBalance.mean()), "YearsAtCompany": yearsatcomp, "YearsInCurrentRole": yearscurrentrole,
                        "YearsSinceLastPromotion": yearslstpromotion, "YearsWithCurrManager": yearswithman}, index=[0])


#tab_model.write(employee)



# Giriş Kontrolleri
if age < 0 or education not in [1, 2, 3, 4, 5]:
    tab_model.error("Geçersiz giriş değerleri. Lütfen girişleri kontrol edin.")
else:
    if col_select2.button(":green[Predict]"):
        with st.spinner('Estimation progress is on'):
            time.sleep(5)
        dff=dff._append(employee, ignore_index=True)
        X, y = hr_data_prep(dff)
        last_index = dff.index[-1]
        last_row = X.iloc[[last_index]]
        prediction = model.predict(last_row)
        if prediction[0]==1:
            st.snow()
            tab_model.image("bye.jpg", width=200)
            tab_model.error("Attention!!! The employee seems attrited and can resign. It is recommended that necessary improvements be made to this employee.")
        else:
            st.balloons()
            tab_model.image("retantion.jpg", width=200)
            tab_model.success("Good news!!! Your employee appears to be retained. Keep supporting")

        #tab_model.success(f"Attrition : {prediction[0]} ")
        tab_model.info("Estimated with a model success rate of 92%.")
        #tab_model.write("Attirition:1 means the employee can resign. It is recommended that necessary improvements be made to this employee.")
        #tab_model.write("Attirition:0 means the employee may not resign.")

