



"""Importing Dependencies

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collections & Processing"""

#load the data from csv file
titanic_data = pd.read_csv('/content/archive.zip')

# printing the first five rows of the dataframe
titanic_data.head()

# no of rows and columns
titanic_data.shape

# getting some information about the data
titanic_data.info()

# check the number of missing value in each columns
titanic_data.isnull().sum()

"""Handling the missing values

"""

#drop the "Cabin" cloumns from the dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis=1)

#replacing the misssing values in "Age" columns with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)

#finding the mode values of "Embarked" columns
print(titanic_data['Embarked'].mode())

#finding the mode values of "Fare" columns
print(titanic_data['Fare'].mode())

print(titanic_data['Fare'].mode()[0])

#replacing the missing values in "Fare" columns with mode values
titanic_data['Fare'].fillna(titanic_data['Fare'].mode()[0],inplace=True)

# check the number of missing value in each columns
titanic_data.isnull().sum()

"""Data Analysis"""

# getting some statistical measures about the data
titanic_data.describe()

#finding the number of people survived or not survived
titanic_data['Survived'].value_counts()

"""Data Visualization"""

sns.set()

# making a count plot for "Survived" columns
sns.countplot(x="Survived", data=titanic_data)

# making a count plot for "Sex" columns
sns.countplot(x="Sex", data=titanic_data)

# number of Survivers Gender wise
sns.countplot(x="Sex", hue="Survived" ,data=titanic_data)

# making a count plot for "Pclass" columns
sns.countplot(x="Pclass", data=titanic_data)

sns.countplot(x="Pclass", hue="Survived" ,data=titanic_data)

sns.countplot(x="Embarked", hue="Survived" ,data=titanic_data)

sns.countplot(x="Parch", hue="Survived" ,data=titanic_data)

sns.countplot(x="SibSp", hue="Survived" ,data=titanic_data)

sns.countplot(x="Age", hue="Survived" ,data=titanic_data)

"""Encoding the Categorical columns"""

titanic_data['Sex'].value_counts()

titanic_data['Embarked'].value_counts()

#converting categorical ccolumns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)

titanic_data.head()

"""Separating features & target"""

X= titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
Y= titanic_data['Survived']

print(X)

print(Y)

"""Spliting the Data into traning data & test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_test.shape, X_train.shape)

"""

Logistic Regression"""

model = LogisticRegression()

# traning the LogisticRegression model with training data
model.fit(X_train, Y_train)

"""Model Evaluation



Accuracy score
"""

#accuracy on training data
X_train_prediction = model.predict(X_train)

print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of traning data  : ',training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)

print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data  : ',test_data_accuracy)

