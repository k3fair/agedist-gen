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

# Set working directory
home =  os.getcwd()[:-4]

# Read in age distribution data
df = pd.read_excel('%sdata/WPP2019_POP_F15_1_ANNUAL_POPULATION_BY_AGE_BOTH_SEXES.xlsx' % home, sheet_name = 'ESTIMATES', header=16)
df = df.drop(df[df.Type=="Label/Separator"].index) # Drop all rows that are blank section separators
df = df.infer_objects() # Correct dtypes for age columns
df = df.round() # Round all population counts to nearest integer

mono_check=np.zeros(df.shape[0])

for i in range(df.shape[0]):
    mono_check[i] = df.iloc[i, -21::].is_monotonic_decreasing #Check which of the age distributions are monotoic decreasing with age

df["monocheck"] = mono_check

# Save 2 sets of age distributions, monotonic decreasing (md) and other
df.loc[df.monocheck==1].to_csv('%sdata/agedists_md.csv' % home) 
df.loc[df.monocheck==0].to_csv('%sdata/agedists_other.csv' % home) 

# Subset to get only country-level distributions for most recent year (2020)
df_countries = df.loc[df.Type=="Country/Area"] 
df_countries_2020 = df_countries.loc[df_countries["Reference date (as of 1 July)"]==2020]
df_countries_2020_other=df_countries_2020.loc[df_countries_2020.monocheck==0]

dec_count=np.zeros(df_countries_2020_other.shape[0])
for i in range(df_countries_2020_other.shape[0]):
    
    dif_list = np.array([j-i for i, j in zip(df_countries_2020_other.iloc[i, -22:-2], df_countries_2020_other.iloc[i, 9:-1])])
    dec_count[i] = len(dif_list[dif_list<=0])
    
    plt.plot(df_countries_2020_other.iloc[i, -22:-1]/df_countries_2020_other.iloc[i, -22:-1].sum())

plt.xticks(rotation=45)
plt.xlabel("Age group")    
plt.show()


plt.hist(dec_count)
plt.show()

for i in range(df_countries_2020_other.shape[0]):
    
    plt.plot(df_countries_2020_other.iloc[i, -22:-1]/df_countries_2020_other.iloc[i, -22:-1].sum())

plt.xticks(rotation=45)
plt.xlabel("Age group")    
plt.show()