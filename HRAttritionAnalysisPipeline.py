import warnings
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('expand_frame_repr', False)


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

    #Salary to Hourly Rate Ratio
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
    df['NEW_MARITALSTATUS_ENVIRONMENTSATISFACTION'] = df['MARITALSTATUS'].map(
        {'Single': 1, 'Married': 2, 'Divorced': 3}) * df['ENVIRONMENTSATISFACTION']

    # Job Role and Monthly Income Interaction
    job_role_income_interaction = df.groupby('JOBROLE')['MONTHLYINCOME'].transform('mean')
    df['NEW_JOBROLE_INCOME_INTERACTION'] = df['JOBROLE'].map(job_role_income_interaction)

    # Years with Current Manager and Salary Interaction
    df['NEW_YEARSWITHCURRENTMANAGER_SALARY_INTERACTION'] = df['YEARSWITHCURRMANAGER'] * df['MONTHLYINCOME']

    # Job Level and Environment Satisfaction Interaction
    df['NEW_JOBLEVEL_ENVIRONMENTSATISFACTION_INTERACTION'] = df['JOBLEVEL'] * df['ENVIRONMENTSATISFACTION']

    cat_cols, cat_but_car, num_cols = grab_col_names(df)

    # Aykırı değerlerin baskılanması
    for col in num_cols:
        if col != "ATTRITION":
            replace_with_thresholds(df, col)

    # Eksik değerlerin doldurulması fonksiyonu

    binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

    for col in binary_cols:
        label_encoder(df, col)

    cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["ATTRITION"]]

    df = one_hot_encoder(df, cat_cols, drop_first=False)

    # log sönüşümü
    for col in num_cols:
        df[col] = np.log1p(df[col])

    # Robust Scaling
    X_scaled = RobustScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["ATTRITION"]
    X = df.drop(["ATTRITION"], axis=1)

    oversampler = SMOTE()
    X_resample, y_resample = oversampler.fit_resample(X, y)

    return X_resample, y_resample


"""
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

adaboost_params = param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)]}

"""
"""catboost_param = param_grid = {'iterations': [100, 200, 300],
                               'learning_rate': [0.01, 0.1, 0.2],
                               'depth': [4, 6, 8],
                               'l2_leaf_reg': [1, 3, 5],
                               'border_count': [32, 64, 128],
                               'thread_count': [2, 4, 8],
                               'colsample_bylevel': [0.7, 0.8, 0.9],
                               'bagging_temperature': [0.2, 0.5, 0.8]}
classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(class_weight={0: 0.2, 1: 0.8}), cart_params),
               ("RF", RandomForestClassifier(class_weight={0: 0.2, 1: 0.8}), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params),
               ('AdaBoost', AdaBoostClassifier(),adaboost_params),
               #('CatBoost',CatBoostClassifier(verbose=False),catboost_param)
               ]

def hyperparameter_optimization(X_resample, y_resample, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X_resample, y_resample, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X_resample, y_resample)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X_resample, y_resample, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models
"""
def voting_classifier(X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(n_neighbors=2)),
                                              ('RF', RandomForestClassifier(class_weight={0: 0.2, 1: 0.8},max_features='auto',min_samples_split=15, n_estimators=300)),
                                              ('LightGBM', LGBMClassifier(learning_rate=0.01, n_estimators=300, verbose=-1))],
                                              voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


def main():
    df = pd.read_csv("C:\HR_data.csv")
    X_resample, y_resample = hr_data_prep(df)
    #best_models = hyperparameter_optimization(X_resample, y_resample)
    voting_clf = voting_classifier(X_resample, y_resample)
    joblib.dump(voting_clf, "model.joblib")
    return voting_clf

# bir python dosyasını işletim sisteminden tetiklemek için yapılan döngü

if __name__ == "__main__":
    print("İşlem başladı")
    main()



