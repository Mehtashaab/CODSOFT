"""

Import Dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

"""Data Collection and Processing

"""

Sales_Prediction = pd.read_csv('/content/advertising.csv')

Sales_Prediction.head()

"""Checking For Nullvalues"""

# getting some information about the data
Sales_Prediction.info()

# check the number of missing value in each columns
Sales_Prediction.isnull().sum()

# number of rows and columns
Sales_Prediction.shape

"""Data Analysis"""

# getting some statistical measures about the data
Sales_Prediction.describe()

sns.set()

# making a count plot for TV Columns
sns.countplot(x="TV",data=Sales_Prediction)

# making a boxplot for Tv Columns
sns.boxplot(x='TV', data=Sales_Prediction)
plt.show()

# making a count plot for Radio Columns
sns.countplot(x="Radio",data=Sales_Prediction)

# making a boxplot for Radio Columns
sns.boxplot(x='Radio', data=Sales_Prediction)
plt.show()

# making a count plot for Newspaper Columns
sns.countplot(x="Newspaper",data=Sales_Prediction)

# making a boxplot for Newspaper Columns
sns.boxplot(x='Newspaper', data=Sales_Prediction)
plt.show()

# making a count plot for Sales Columns
sns.countplot(x="Sales",data=Sales_Prediction)

# making a boxplot for Sales Columns
sns.boxplot(x='Sales', data=Sales_Prediction)
plt.show()

sns.pairplot(Sales_Prediction, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

sns.heatmap(Sales_Prediction.corr(), cmap="Greens", annot = True)
plt.show()

X=Sales_Prediction[["TV","Radio","Newspaper"]]
Y=Sales_Prediction[["Sales"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_test.shape, X_train.shape)

model = LinearRegression()

# traning the LinearRegression model with training data
model=LinearRegression(fit_intercept=True)
model.fit(X_train, Y_train)

#accuracy on training data
X_test_prediction = model.predict(X_test)

print(X_test_prediction)

#accuracy on training data
X_train_prediction = model.predict(X_train)

print(X_train_prediction)



"""Evaluation"""

mse= mean_squared_error(Y_test,Y_predict)
mse

rmse = mean_squared_error(Y_test, Y_predict, squared = False)
rmse

r2 = r2_score(Y_test, Y_predict)
r2



