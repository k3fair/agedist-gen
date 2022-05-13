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
df = pd.read_csv('%sdata/agedists_countries2020_other.csv' % home)

df_wassdist = np.zeros(df.shape[0])

# for i in np.arange(0,df.shape[0]):

# target_dist = df.iloc[i, -22:-1] #For now just do the check on the very first entry    
# n_groups = len(target_dist) # Set number of groups to match age classes in data
# target_dist_cum = np.cumsum(target_dist/target_dist.sum())

numselect=125
target_dist = df.iloc[numselect, -22:-1] #For now just do the check on the very first entry
n_groups = len(target_dist) # Set number of groups to match age classes in data
target_dist_cum = np.cumsum(target_dist/target_dist.sum())


plt.figure(0, figsize=(8,4))

plt.bar(height=target_dist/target_dist.sum(), x=target_dist.index)
plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("P(age==x)")
plt.title(f"Age distribution: {df.iloc[numselect,3]}")

plt.tight_layout()
plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
plt.show()

# plt.figure(0, figsize=(5,5))

# plt.plot(target_dist)
# plt.xticks(rotation=45)
# plt.xlabel("Age group")
# plt.ylabel("P(age==x)")
# plt.title("Age distribution")

# plt.tight_layout()
# plt.show()

# Generate array of dummy data matching the length of our target distribution
xdat = np.arange(0, len(target_dist_cum))

xdat = np.asarray(xdat)
ydat = np.asarray(target_dist_cum)

# plt.plot(xdat, ydat, 'o')
# plt.show()

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

fit_y_old = mmfcn(xdat, fit_A, fit_B, fit_k)
fit_y_old[fit_y_old>1] = 1 

# plt.plot(xdat, ydat, 'o', label='data')
# plt.plot(xdat, fit_y_old, '-', label='fit')
# plt.legend()
# plt.show()

fit_y_noncum = fit_y_old.copy()
fit_y_noncum[1:] -= fit_y_noncum[:-1].copy()

# plt.plot(xdat, target_dist/target_dist.sum() , 'o', label='data')
# plt.plot(xdat, fit_y_noncum , '-', label='fit')
# plt.legend()
# plt.show()

# print(wasserstein_distance(target_dist_cum, fit_y_old))

startbin=10

# Define the decay equation we're using for fitting
def decayfcn(x, A, B, k):
    
    y = A*np.ones(len(x))
    y[startbin:] = A*np.exp(-B*((x[startbin:] - x[startbin])**k))
    return y

# Fit curve
ydat = np.asarray(target_dist/target_dist.sum())
parameters, covariance = curve_fit(decayfcn, xdat, ydat)

fit_A = parameters[0]
fit_B = parameters[1]
fit_k = parameters[2]

fit_y = decayfcn(xdat, fit_A, fit_B, fit_k)
# fit_y[fit_y>1] = 1 

wass_dist = wasserstein_distance(target_dist/target_dist.sum(), fit_y)

print(wass_dist)

# if wass_dist > 0.01:

plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, ydat, 'o', label='data')
plt.plot(xdat, fit_y, '-', label='fit')
plt.plot(xdat, fit_y_noncum , '--', label='fit (cum-based)')

plt.subplot(122)
plt.plot(xdat, np.cumsum(ydat), 'o', label='data')
plt.plot(xdat, np.cumsum(fit_y), '-', label='fit')
plt.plot(xdat, fit_y_old, '--', label='fit (cum-based)')
plt.legend()
plt.show()

plt.figure(0, figsize=(8,4))

plt.bar(x=target_dist.index, height=target_dist/target_dist.sum(), label='data')
plt.plot(xdat, fit_y, 'o-', color="orange", label='fit')
plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("P(age==x)")
plt.legend()
plt.title(f"Age distribution: {df.iloc[numselect,3]}")

plt.tight_layout()
plt.savefig(f'agedist_fitted_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
plt.show()

if abs(fit_y.sum()-1) > 0.01:
    print("Fitted probabilties aren't summing nicely")

# df_wassdist[i] = wass_dist
    
# plt.figure(2, figsize=(10,5))    
# plt.hist(df_wassdist)
# plt.show()

