# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:34:16 2022

@author: adheeb
"""

import numpy as np
import matplotlib.pyplot as plt

array = np.loadtxt("inputdata6.csv", delimiter=",", skiprows=1)
rainfall = np.array(array[:, 0]).reshape((-1, 1))
productivity = np.array(array[:, 1])
plt.scatter(rainfall, productivity)
