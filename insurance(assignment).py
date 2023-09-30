# importing python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reding data_csv file
data=pd.read_csv('insurance (1).csv')

#EDA
data.shape
data.head()
data.tail()
data.columns
data.info()
data.describe()

#################

# COMPARING AGE , BMI WITH CHARGES
# ABOVE MENTIONED COLUMNS ARE INTEGER, SO WE HAVE TO USE SCATTER FOR COMPARISON

sns.scatterplot(x=data['age'],y=data['charges'])
sns.scatterplot(x=data['bmi'],y=data['charges'])
###############################

# HERE DATA IS TOO DISCRETE SO WE ARE UNABLE TO FETCH INFORMATION
# FOR LABEL VS INTEGER , WE HAVE TO USE BOXPLOT
#Gender vs Charges

sns.boxplot(x=data['sex'],y=data['charges'])
#Children vs Charges

sns.boxplot(x=data['children'],y=data['charges'])
###############################
#Smoker vs Charges

sns.boxplot(x=data['smoker'],y=data['charges'])
#Region vs Charges

sns.boxplot(x=data['region'],y=data['charges'])
############

columns=['sex','smoker', 'region']
for column in columns:
    print(data[column].unique())
    print(data[column].value_counts())
 #CONVERTING LABELS INTO NUMERICAL FORM WITH THE HELP OF LABEL ENCODER
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
for column in columns:
   data[column]= encoder.fit_transform(data[column])
data.columns
data.info()
data.describe()
data.isna().sum() # checking for numll values 

################
# SEGREEGATE DATA INTO TRAINING AND TESTING
# WE WILL USE TRAIN_TEST MOUDLE FROM SKLEARN.MODEL_SELECTION
# SEPARATE DATA INTO INPUT AND OUTPUT 
x=data.drop(['charges'],axis=1)
y=data['charges']
###############################
# NOW,WE HAVE TO CALL REGRESSOR FOR OUR PREDICTIONS
# USE LINEAR REGRESSION FROM SKLEARN.LINEAR MODAL
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# NOW,WE HAVE TO CALL REGRESSOR FOR OUR PREDICTIONS
# and fit multiple REGRESSION model FROM SKLEARN.LINEAR MODAL
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# making prediction using the model
y_pred = regressor.predict(x_test)

# CHECK ACCURACY OF THE MODAL
# USE METRICS FROM SKLEARN TO CHECK ERROR

from sklearn import metrics 
np.sqrt(metrics.mean_squared_error(y_test, y_pred)) #5663.358417062193
metrics.mean_absolute_error(y_test, y_pred)         #3998.271540886974
metrics.r2_score(y_test,y_pred)                     #0.7962732059725786

###################END
 
 

















