# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:50:39 2021

@author: Lenovo
"""

# Importing Librarues
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# Training decision tree regression model on whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)


# Predicting a new result
print(regressor.predict([[6.5]]))


# Visualising Decision Tree Regression results("HIGHER RESOLUTION")
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.show()