import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import Lasso


df_train = pd.read_csv("train.csv")
df_train.columns
df_train.head()
df_train.describe()

pd.set_option('display.float_format',lambda x: '%.2f' % x)
df_train['price'].describe()

sns.distplot(df_train['price'])

#skewness and kurtosis
print("Skewness: %f" % df_train['price'].skew())
print("Kurtosis: %f" % df_train['price'].kurt())

# df_train['price'].skew()

# sns.histplot(df_train['price'],bins=100,kde=True,color='red')

#I found that my data has really HIGH Kurtosis and it is right skewed
#Let me try to apply log function to get this smooth

df_train['price_transformed'] = np.log(df_train['price'])
# df_train.head(2) 

sns.distplot(df_train['price_transformed'])

print("Skewness: %f" % df_train['price_transformed'].skew())
print("Kurtosis: %f" % df_train['price_transformed'].kurt())

#Excellent. Now I learnt how why to apply log transformation

#Trying to understand what indep. varialble are influencing Price

# data1 = pd.concat([df_train['price'],df_train['model_year']],axis=1)
# data1
# data1.plot.scatter(x='model_year',y='price',ylim=(0,1000000))
# df_train.info()

#sns.pairplot(df_train)

#sns.heatmap(df_train.corr(),annot=True)

#checking missing data %
total_missing = df_train.isnull().sum().sort_values(ascending=False)
total_missing
percent_missing = ((df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False))*100
percent_missing

df_train['clean_title'].fillna('No', inplace=True)

percent_missing = ((df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False))*100
percent_missing


#I am going to delete all the missing data for my easiness :)
df_train = df_train.dropna()
df_train.isnull().sum()

df_train.info()

#dropping 'id' column
df_train = df_train.drop(['id'],axis=1)

#removing duplicates if any
df_train.drop_duplicates(inplace=True)

#Lets do some data preprocessing

df_train.head(2)

#Extract information from 'engine' column

import re
df_train['engine_new'] = df_train['engine'].str.extract(r'(\d+\.\d+)L')[0].astype(float)
df_train.head(1)
df_train.isnull().sum()


df_train2 = df_train.copy()
df_train2.isnull().sum()
df_train2.info()

#Model Age from 'Model year' column

import datetime
current_year = datetime.datetime.now().year
current_year

df_train2['model_age'] = current_year - df_train2['model_year']

df_train2.head(2)

df_train2['transmission'].value_counts()


def assign_group(description):
    if 'Automatic' in description:
        return 'Automatic'
    elif 'Manual' in description:
        return 'Manual'
    elif 'M/T' in description:
        return 'Manual'
    elif 'A/T' in description:
        return 'Automatic'
    elif 'Dual Shift' in description:
        return 'Semi-Automatic'
    elif 'CVT' in description:
        return 'Automatic'
    else:
        return 'Not Clear'
    
df_train2['transmission_category'] = df_train2['transmission'].apply(assign_group)
df_train2.transmission_category.value_counts()
df_train2.isnull().sum()

df_train2['accident'].value_counts()
df_train2['clean_title'].value_counts()
df_train2['transmission_category'].value_counts()
df_train2['fuel_type'].value_counts()


def mapping_columns(x):
    df_train2['accident'] = df_train2['accident'].replace({
        'At least 1 accident or damage reported' : 1,
        'None reported' : 0
    })
    df_train2['clean_title'] = df_train2['clean_title'].replace({
        'Yes' : 1,
        'No' : 0
    })
    df_train2['transmission_category'] = df_train2['transmission_category'].replace({
        'Automatic' : 1,
        'Semi-Automatic' : 2,
        'Manual' : 3,
        'Not Clear' : 4
    })
    return x 

df_train2 = mapping_columns(df_train2)
df_train2.head()
    
df_train2['accident'].value_counts()
df_train2['clean_title'].value_counts()
df_train2['transmission_category'].value_counts()

df_train2['brand'].value_counts()
df_train2['fuel_type'].value_counts()
df_train2['ext_col'].value_counts()
df_train2['int_col'].value_counts()


#Encoding
categorical_columns = ['brand', 'fuel_type', 'ext_col', 'int_col']
lb = LabelEncoder()



# Encoding categorical variables
categorical_columns = ['brand', 'fuel_type', 'ext_col', 'int_col']
lb = LabelEncoder()

for col in categorical_columns:
    if col in df_train2.columns:
        df_train2[col] = lb.fit_transform(df_train2[col])

df_train2.head(2)      

#dropping categorical data

df_train3 = df_train2.drop(['model','model_year','engine','transmission'],axis=1)
df_train3.head()
df_train3.info()

#Visualize
plt.figure(figsize=(12,12))
sns.heatmap(df_train3.corr(),annot=True,fmt='.2f')

df_train3.info()
df_train3.dropna(inplace=True)
df_train3.isnull().sum()

df_final = df_train3.drop(['price'],axis=1)
df_final.head(1)
X = df_final.drop(['price_transformed'],axis=1)
y = df_final['price_transformed']

X.shape
y.shape

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
X.head(1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
X_train.shape
y_train.shape

#Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
model_lr.score(X_train,y_train)

y_pred_lr = model_lr.predict(X_test)
#model_lr.score(y_test,y_pred)

from sklearn import metrics
rmse = np.sqrt(metrics.mean_absolute_error(y_test,y_pred_lr))
print(f'Linear Regression RMSE: {rmse:.2f}')


#Decision Tree
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train,y_train)
model_dt.score(X_train,y_train)

y_pred_dt = model_dt.predict(X_test)
rmse = np.sqrt(metrics.mean_absolute_error(y_test,y_pred_dt))
print(f'Linear Regression RMSE: {rmse:.2f}')

#Random Forest
model_rf = RandomForestRegressor(n_estimators=100,random_state=10)
model_rf.fit(X_train,y_train)
model_rf.score(X_train,y_train)

y_pred_rf = model_rf.predict(X_test)
rmse = np.sqrt(metrics.mean_absolute_error(y_test,y_pred_rf))
print(f'Linear Regression RMSE: {rmse:.2f}')




