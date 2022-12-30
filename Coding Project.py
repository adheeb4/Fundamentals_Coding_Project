# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:34:16 2022

@author: adheeb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

array = np.loadtxt("inputdata6.csv", delimiter=",", skiprows=1)
rainfall = np.array(array[:, 0]).reshape((-1, 1))
productivity = np.array(array[:, 1])
plt.scatter(rainfall, productivity)
model = LinearRegression().fit(rainfall, productivity)
slope = model.coef_
intercept = model.intercept_
plt.plot(rainfall, slope*rainfall+intercept, color='black')
prediction = model.predict([[245]])
