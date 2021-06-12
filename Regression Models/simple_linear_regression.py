# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:10:59 2021

@author: Lenovo
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing Datasets
dataset = pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values



#  Spliting Dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=1)


# Training Simple Linear Regression Model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# Predicting the test set result
y_pred = regressor.predict(x_test)


# Visualising Training set resukt
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


# Visualising test set result
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))


# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)