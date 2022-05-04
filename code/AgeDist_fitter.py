#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:41:27 2022

@author: kfair
"""

#Load in required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance

# Set working directory
home =  os.getcwd()[:-4]

# Read in age distribution data for distributions which are not monotonic decreasing
df = pd.read_csv('%sdata/agedists_md.csv' % home)

target_dist = df.iloc[100, -22:-1] #For now just do the check on the very first entry
n_groups = len(target_dist) # Set number of groups to match age classes in data
target_dist_cum = np.cumsum(target_dist/target_dist.sum())

plt.figure(0, figsize=(5,5))

plt.plot(target_dist)
plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("P(age==x)")
plt.title("Age distribution")

plt.tight_layout()
plt.show()

# Generate array of dummy data matching the length of our target distribution
xdat = np.arange(0, len(target_dist_cum))

xdat = np.asarray(xdat)
ydat = np.asarray(target_dist_cum)
plt.plot(xdat, ydat, 'o')

# Define the michaelis-menten/monod style equation we're using for fitting
def mmfcn(x, A, B, k):
    y = A + ((B*x)/(x+k))
    return y

# Fit curve
parameters, covariance = curve_fit(mmfcn, xdat, ydat)

fit_A = parameters[0]
fit_B = parameters[1]
fit_k = parameters[2]
# print(fit_A)
# print(fit_B)
# print(fit_k)

fit_y = mmfcn(xdat, fit_A, fit_B, fit_k)
fit_y[fit_y>1] = 1 

plt.plot(xdat, ydat, 'o', label='data')
plt.plot(xdat, fit_y, '-', label='fit')
plt.legend()
plt.show()

fit_y_noncum = fit_y.copy()
fit_y_noncum[1:] -= fit_y_noncum[:-1].copy()

plt.plot(xdat, target_dist/target_dist.sum() , 'o', label='data')
plt.plot(xdat, fit_y_noncum , '-', label='fit')
plt.legend()
plt.show()

print(wasserstein_distance(target_dist_cum, fit_y))