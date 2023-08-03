
"""
Importing Dependencies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Replace 'input_file_path' with the path to your input file.

input_file_path = "/content/movies.dat"

# Replace 'output_file_path' with the desired path for the output .csv file.


output_file_path = "output_file.csv"

# Read the .dat file into a DataFrame using a specific encoding.


Movie_rating = pd.read_csv(input_file_path, encoding='latin1',sep = '::', engine='python',header=None)

# Write the DataFrame to a .csv file.


Movie_rating.to_csv(output_file_path, index=False)
Movie_rating.head(20)

Movie_rating.info()

# Giving attribute names
Movie_rating.columns = ['Movie_Id','Movie_Title','Genre']
Movie_rating.dropna(inplace=True)
Movie_rating.head(10)

Movie_rating.tail(10)

Movie_rating.shape

Movie_rating.info()

# Checking missing values
Movie_rating.isnull().sum()

"""Loading Rating dataset"""

# Replace 'input_file_path' with the path to your input file.
input_file_path = "/content/ratings.dat"

# Replace 'output_file_path' with the desired path for the output .csv file.
output_file_path = "ratings.csv"

# Read the .dat file into a DataFrame using a specific encoding.
ratings_data = pd.read_csv(input_file_path, encoding='latin1',sep = '::', engine='python',header=None)

# Write the DataFrame to a .csv file.
ratings_data.to_csv(output_file_path, index=False)
ratings_data.head(20)

# Giving attribute names
ratings_data.columns = ['Id', 'Movie_Id', 'Rating','Timestamp']
ratings_data.dropna(inplace=True)
ratings_data.head(10)

ratings_data.tail(10)

ratings_data.shape

ratings_data.info()

# Check missing values
ratings_data.isnull().sum()

"""Loading Users dataset"""

# Replace 'input_file_path' with the path to your input file.
input_file_path = "/content/users.dat"

# Replace 'output_file_path' with the desired path for the output .csv file.
output_file_path = "Users.csv"

# Read the .dat file into a DataFrame using a specific encoding.
Users_data = pd.read_csv(input_file_path, encoding='latin1',sep = '::', engine='python',header=None)

# Write the DataFrame to a .csv file.
Users_data.to_csv(output_file_path, index=False)
Users_data.head(20)

# Giving attribute names
Users_data.columns = ['User_Id','Gender','Age','Occupation','Zip-code']
Users_data.dropna(inplace=True)
Users_data.head(10)

Users_data.tail(10)

# Changing string values to integer
Users_data['Gender'].replace({'M':0, 'F':1}, inplace = True)
Users_data
Users_data

Users_data.shape

Users_data.info()

# Checking missing values
Users_data.isnull().sum()

"""Finding Relation"""

# Distinct values of MovieIds
unique_counts = ratings_data['Movie_Id'].nunique()
print('Movie_Id:',unique_counts)

# Min values of MovieIds
min_rating = ratings_data['Movie_Id'].min()
print("'{}':{}".format('Movie_Id',min_rating))

# Max values of MovieIds
max_rating = ratings_data['Movie_Id'].max()
print("'{}':{}".format('Movie_Id',max_rating))

# Distinct values of Ids
unique_counts_ids = ratings_data['Id'].nunique()
print('Id:',unique_counts_ids)

# Min values of Ids
min_rating_id = ratings_data['Id'].min()
print("'{}':{}".format('Id',min_rating))

# Max values of Ids
max_rating_id = ratings_data['Id'].max()
print("'{}':{}".format('Id',max_rating_id))

"""Merging dataset"""

# Merge 'Movie', 'ratings' and 'Users' dataframe on the basis of common columns
merged_data = pd.merge(ratings_data, Users_data, left_on='Id', right_on='User_Id')
merged_data = pd.merge(ratings_data, Movie_rating, on='Movie_Id')

merged_data.head()

merged_data.shape

merged_data.info()

merged_data.head(10)

merged_data.tail(10)

# Checking missing values
merged_data.isnull().sum()

# To calculate count of users using MovieId and Rating
rating_counts = merged_data.groupby(['Movie_Id','Rating']).size().reset_index(name = 'UserCount')

# To get movies which has more ratings than 100
filter_data = rating_counts[rating_counts['UserCount']>=100]

filter_data = pd.merge(filter_data,merged_data[['Movie_Id','Rating','Genre']])
filter_data

filter_data.shape

# After combining all the datasets
data = pd.concat([Movie_rating, ratings_data, Users_data], axis=1)
data.head(20)

data.info()

"""Data Visulization"""

# To visualize overall ratingby users
filter_data['Rating'].value_counts().plot(kind='bar',alpha = 0.7,figsize=(10, 10))
plt.show()

# Create histogram for combined datasets
data.Age.plot.hist(bins = 15)
plt.title('Distribution of age')
plt.ylabel('count of users')
plt.xlabel('age')

# Count the occurrences of each rating
ratings_counts = ratings_data['Rating'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(ratings_counts, labels=ratings_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Movie Ratings Distribution')
plt.axis('equal')
plt.show()

# Histogram for movie ratings
plt.figure(figsize=(8, 6))
plt.hist(ratings_data['Rating'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()

"""Model Building"""

first_700 = filter_data[700:]
first_700.dropna(inplace=True)

# Splitting of data into training and testing set
x = first_700.drop(['Genre'],axis=1)
y = first_700['Rating']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

print(x.shape, x_train.shape, x_test.shape)

"""Model Training"""

model = LogisticRegression()

x_encoded = pd.get_dummies(x)

model.fit(x_train, y_train)

"""Model Evaluation"""

# Mean Squared Error
y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print('Mean Squared Error:',mse)

# Root Mean Squared Error
rmse = np.sqrt(mse)
print('Root Mean Squared Error:',rmse)

# Training accuracy
log = round(model.score(x_train, y_train) * 100, 2)
print(log)

# Testing accuracy
log = round(model.score(x_test, y_test) * 100, 2)
print(log)

