
"""
Importing Dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# importing the dataset to pandas dataframe
credit_card_fraud = pd.read_csv("/content/creditcard.csv")

credit_card_fraud.head()

credit_card_fraud.tail()

#dataset information
credit_card_fraud.info()

#checking the number of missing values in rach columns
credit_card_fraud.isnull().sum()

credit_card_fraud['V3'].fillna(credit_card_fraud['V3'].mean(),inplace=True)

credit_card_fraud['V4'].fillna(credit_card_fraud['V4'].mean(),inplace=True)

credit_card_fraud['V5'].fillna(credit_card_fraud['V5'].mean(),inplace=True)

credit_card_fraud['V6'].fillna(credit_card_fraud['V6'].mean(),inplace=True)

credit_card_fraud['V7'].fillna(credit_card_fraud['V7'].mean(),inplace=True)
credit_card_fraud['V8'].fillna(credit_card_fraud['V8'].mean(),inplace=True)
credit_card_fraud['V9'].fillna(credit_card_fraud['V9'].mean(),inplace=True)
credit_card_fraud['V10'].fillna(credit_card_fraud['V10'].mean(),inplace=True)
credit_card_fraud['V11'].fillna(credit_card_fraud['V11'].mean(),inplace=True)
credit_card_fraud['V12'].fillna(credit_card_fraud['V12'].mean(),inplace=True)
credit_card_fraud['V13'].fillna(credit_card_fraud['V13'].mean(),inplace=True)
credit_card_fraud['V14'].fillna(credit_card_fraud['V14'].mean(),inplace=True)
credit_card_fraud['V15'].fillna(credit_card_fraud['V15'].mean(),inplace=True)
credit_card_fraud['V16'].fillna(credit_card_fraud['V16'].mean(),inplace=True)
credit_card_fraud['V17'].fillna(credit_card_fraud['V17'].mean(),inplace=True)
credit_card_fraud['V18'].fillna(credit_card_fraud['V18'].mean(),inplace=True)
credit_card_fraud['V19'].fillna(credit_card_fraud['V19'].mean(),inplace=True)
credit_card_fraud['V20'].fillna(credit_card_fraud['V20'].mean(),inplace=True)
credit_card_fraud['V21'].fillna(credit_card_fraud['V21'].mean(),inplace=True)
credit_card_fraud['V22'].fillna(credit_card_fraud['V22'].mean(),inplace=True)
credit_card_fraud['V23'].fillna(credit_card_fraud['V23'].mean(),inplace=True)
credit_card_fraud['V24'].fillna(credit_card_fraud['V24'].mean(),inplace=True)
credit_card_fraud['V25'].fillna(credit_card_fraud['V25'].mean(),inplace=True)
credit_card_fraud['V26'].fillna(credit_card_fraud['V26'].mean(),inplace=True)
credit_card_fraud['V27'].fillna(credit_card_fraud['V27'].mean(),inplace=True)
credit_card_fraud['V28'].fillna(credit_card_fraud['V28'].mean(),inplace=True)

#finding the mode values of "Amount" columns
print(credit_card_fraud['Amount'].mode())

#finding the mode values of "Class" columns
print(credit_card_fraud['Class'].mode())

print(credit_card_fraud['Class'].mode()[0])

#replacing the missing values in "Class" columns with mode values
credit_card_fraud['Class'].fillna(credit_card_fraud['Class'].mode()[0],inplace=True)

print(credit_card_fraud['Amount'].mode()[0])

#replacing the missing values in "Amount" columns with mode values
credit_card_fraud['Amount'].fillna(credit_card_fraud['Amount'].mode()[0],inplace=True)

credit_card_fraud.isnull().sum()

# distribution of logic Transaction and Fraudulant transaction
credit_card_fraud['Class'].value_counts()

"""This dataset is highly unbalanced

0 stands for Normal Transaction

1 stands for Fraudulent Transaction
"""

# seperate data for analysis
real = credit_card_fraud[credit_card_fraud.Class == 0]
fraud = credit_card_fraud[credit_card_fraud.Class == 1]

print(real.shape)
print(fraud.shape)

# statistical measures of the data
real.Amount.describe()

fraud.Amount.describe()

# compare the values for both transaction
credit_card_fraud.groupby('Class').mean()

"""Under sampling

Build a samle dataset containing similar distribution of transaction and fraudulent transaction
"""

real_sample = real.sample(n=38)

"""concatenating two dataframe"""

new_dataset = pd.concat([real_sample, fraud],axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

"""Split the data into features and target"""

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(X)

print(Y)

"""To split the data into Training data and Testing data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Training Model

Logistic Regression
"""

model = LogisticRegression()

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

"""Evaluation"""

#accuracy on training data
X_train_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(X_train_prediction, Y_test)

print('Accuracy on Traning data : ',training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on test data : ', test_data_accuracy)


