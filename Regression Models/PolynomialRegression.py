# -*- coding: utf-8 -*-
#"""
#Created on Sun Jun  6 10:14:30 2021

#@author: Lenovo
#"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Datasets
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Training Linear Regression Model on Whole Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# Training Polynomial regression Model on whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)


# Visualising Linear Regression Results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()


# Visualising Polynomial Regression Results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()


# Pridicting a new result with linear regression
print(lin_reg.predict([[6.5]]))


# Pridicting a new result with Polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))