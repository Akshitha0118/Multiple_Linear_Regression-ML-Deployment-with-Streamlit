# Import libraries 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# read the dataset 
dataset = pd.read_csv(r'C:\Users\ADMIN\Downloads\Investment.csv')

# divide the X & y variables 
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:,4].values


# LabelEncoder to convert the categorical values to numericals values 
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,-1]=labelencoder_x.fit_transform(X[:,-1])

# split the dataset to train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

# linear regression for ml model 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predict the model
y_pred=regressor.predict(X_test)

# slope 
m_slope = regressor.coef_
print(m_slope)

# intercept / constant 
c = regressor.intercept_
print(c)

# add a new column with constant value [56534]
X = np.append(arr=np.full((50,1),56534).astype(int),values=X,axis=1)



# statsmodels.api for interface to various models
import statsmodels.api as sm
X = pd.DataFrame(X).astype(float)
y = pd.Series(y).astype(float)
X = sm.add_constant(X)
X_opt = X.iloc[:, [0,1,2,3,4]]
# ordinary least square
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()


# removing the 4th attribute p- value > 0.05
X = pd.DataFrame(X).astype(float)
y = pd.Series(y).astype(float)
X = sm.add_constant(X)
X_opt = X.iloc[:, [0,1,2,3]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()


# removing the 3rd attribute p- value > 0.05
X = pd.DataFrame(X).astype(float)
y = pd.Series(y).astype(float)
X = sm.add_constant(X)
X_opt = X.iloc[:, [0,1,2]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# removing the 2rd attribute p- value > 0.05
X = pd.DataFrame(X).astype(float)
y = pd.Series(y).astype(float)
X = sm.add_constant(X)
X_opt = X.iloc[:, [0,1]]
regressor_OLS = sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# bias score
bias = regressor.score(X_train, y_train)
bias

# variance score
variance = regressor.score(X_test, y_test)
variance

import pickle 
filename='Multiple_Linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print('model has been as pickled')