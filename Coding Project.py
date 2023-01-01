# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:34:16 2022

@author: adheeb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def read_data(filename):
    """This function takes filename as argument and reads
    file into 2d numpy array and returns the array"""
    array = np.loadtxt(filename, delimiter=",", skiprows=1)
    return array


def plot(x, y, m, b):
    """This function plots scatter plot and linear regression
    line with the given arguments"""
    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, m*x+b, color="orange")
    plt.title("Correlation between Rainfall and Productivity", size=18, pad=45)
    plt.text(20, 0.22, "Predicted Value \nFor X = 245, Y = {:f}"
             .format(float(prediction)), fontsize=12, color="purple")
    plt.xlabel("Rainfall Precipitation (in mm per year)", size=12)
    plt.ylabel("Productivity coefficient", size=12)
    plt.show()


def linear_relation(lr, x):
    """This function takes regression model as argument, finds and returns
    slope, intercept and prediction for the given linear regression model."""
    m = lr.coef_
    b = lr.intercept_
    predict = lr.predict([[x]])
    return m, b, predict


# assigning te value returned from function read_data() to variable data
data = read_data("inputdata6.csv")
# making a numpy array with first column in data and reshaping it to 2d array
rainfall = np.array(data[:, 0]).reshape((-1, 1))
# making a numpy array having the values of second column in data
productivity = np.array(data[:, 1])
# making a linear regression model using the values
model = LinearRegression().fit(rainfall, productivity)
# calling the linear_relation function
slope, intercept, prediction = linear_relation(model, 245)
# calling the plot function
plot(rainfall, productivity, slope, intercept)
