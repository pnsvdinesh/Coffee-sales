# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# Load the dataset
data = pd.read_csv(r"C:\Users\ASUS\Desktop\UMIP272429 P2-Files\index.csv")
data.describe()
data.info()
data.head()
data.tail()

data.isnull()
data.isnull().sum()
data.duplicated().sum()
data.describe().T
data.loc[:,['cash_type','card','coffee_name']].describe().T
data[data['card'].isnull()]['cash_type'].value_counts()
# All transactions with null card information are from cash users 

data['cash_type'].value_counts(normalize=True)
# 92% of the trasancations are  from the card users

# Let's us know which coffee is more popular 
pd.DataFrame(data['coffee_name'].value_counts(normalize=True).sort_values(ascending=False).round(4)*100)
# Americano with Milk having 23.65 proportion and Latte having 21.45 proportion. These two coffees are more popular.

# Convert date and datetime to datetime format
data['date']=pd.to_datetime(data['date'])
data['datetime']=pd.to_datetime(data['datetime'])

# Create Month, Hour, Weekday Columns for further data analysis
data['Month']=data['date'].dt.strftime('%Y-%m')
data['day']=data['date'].dt.strftime('%w')
data['hour']=data['datetime'].dt.strftime('%H')

data.info()
data.head()

# Let's us know the time range of this dataset
[data['date'].min(),data['date'].max()]
# The Time range of this dataset is 2024-03-01 to 2024-07-31

# Groupby the coffee products to know the total revenue 
Data= data.groupby(['coffee_name']).sum(['money']).reset_index().sort_values(by='money', ascending = False)

# plotting the barplot of coffee products to know highest and lowest revenue products
plt.figure(figsize=(10,4))
ax= sns.barplot(data = Data, x= 'money', y='coffee_name')
ax.bar_label(ax.containers[0],fontsize= 8)
plt.xlabel('Revenue')
# The Latte is the product generated highest revenue

#  Monthly_data Sales Analysis#
Data2 = data.groupby(['coffee_name','Month']).count()['date'].reset_index().rename(columns={'date':'count'}).pivot(index='Month', columns='coffee_name',values='count').reset_index()
Data2.describe().T.loc[:,['min','max']]

plt.figure(figsize=(12,6))
sns.lineplot(data=Data2)
plt.legend(loc='upper left')
plt.xticks(range(len(Data2['Month'])), Data2['Month'], size= 'small')

# Weekday_data Sales Analysis
Data3= data.groupby(['day']).count()['date'].reset_index().rename(columns={'date':'count'})

plt.figure(figsize=(12,6))
sns.barplot(data=Data3,x='day',y='count',color='steelblue')
plt.xticks(range(len(Data3['day'])),['Sun','Mon','Tue','Wed','Thur','Fri','Sat'],size='small')
# The Tuesday has highest sales of the week

# Daily_data Sales Analysis
Data4 =data.groupby(['coffee_name','date']).count()['datetime'].reset_index().reset_index().rename(columns={'datetime':'count'}).pivot(index='date',columns='coffee_name',values='count').reset_index().fillna(0)
# Let's us know how many of each products can be sold
Data4.iloc[:,1:].describe().T.loc[:,['min','max']]

#  Hourly_data Sales Analysis
Data5 =data.groupby(['hour']).count()['date'].reset_index().rename(columns={'date':'count'})

sns.barplot(data=Data5,x='hour',y='count',color='red')
# Only 10:00 am and 7:00 pm are peak hours within each day 

# Hourly-Sales by coffee product
Data6 =data.groupby(['hour','coffee_name']).count()['date'].reset_index().rename(columns={'date':'count'}).pivot(index='hour',columns='coffee_name',values='count').fillna(0).reset_index()

# Drop the unnecessary columns
data1 = data.drop(columns=['date','datetime','card'])

# Convert the categorical column  to numerical columns
# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = data1.iloc[:, 0:7]

y = data1['cash_type']

data1.columns

X['coffee_name']= labelencoder.fit_transform(X['coffee_name'])
X['Month'] = labelencoder.fit_transform(X['Month'])

### label encode y ###
y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
data1_new = pd.concat([X, y], axis =1)

## rename column name
data1_new.columns
data1_new = data1_new.rename(columns={0:'Cash_Type'})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Define features and target variable
X = data1_new[['coffee_name','Month','day','hour','Cash_Type']]
y = data1_new['money']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model 
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()






