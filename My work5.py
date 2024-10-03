import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("train.csv")
df_train.columns

pd.set_option('display.float_format',lambda x: '%.2f' % x)
df_train['price'].describe()

sns.distplot(df_train['price'])

#skewness and kurtosis
print("Skewness: %f" % df_train['price'].skew())
print("Kurtosis: %f" % df_train['price'].kurt())

df_train['price'].skew()

sns.histplot(df_train['price'],bins=100,kde=True,color='red')

#I found that my data has really HIGH Kurtosis and it is right skewed
#Let me try to apply log function to get this smooth

df_train['price_transformed'] = np.log(df_train['price'])
df_train.head(2) 

sns.distplot(df_train['price_transformed'])

print("Skewness: %f" % df_train['price_transformed'].skew())
print("Kurtosis: %f" % df_train['price_transformed'].kurt())

#Excellent. Now I learnt how why to apply log transformation

#Trying to understand what indep varialble are influencing Price

data1 = pd.concat([df_train['price'],df_train['model_year']],axis=1)
data1
data1.plot.scatter(x='model_year',y='price',ylim=(0,1000000))
df_train.info()

sns.pairplot(df_train)