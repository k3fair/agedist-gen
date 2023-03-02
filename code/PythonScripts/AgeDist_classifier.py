#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:41:27 2022

Code containg functions needed to classify age distributions based on whether they are monotonic decreasing (and thus can be generated using model 1).
NB: If using a different source of age distribution data, this script will need to be modified.

@author: Kathyrn R. Fair
"""

#Load in required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# import seaborn as sns

# Set working directory
home =  os.getcwd()[:-4]

# Read in age distribution data
df = pd.read_excel(f'{home}data/WPP2019_POP_F15_1_ANNUAL_POPULATION_BY_AGE_BOTH_SEXES.xlsx', sheet_name = 'ESTIMATES', header=16)
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

# Select a year to focus on
select_year = 2019

# Subset to get only country-level distributions for selected year
df_countries = df.loc[df.Type=="Country/Area"] 
df_countries_year = df_countries.loc[df_countries["Reference date (as of 1 July)"]==select_year]
df_countries_year_other=df_countries_year.loc[df_countries_year.monocheck==0]
df_countries_year_md=df_countries_year.loc[df_countries_year.monocheck==1]

# # Generate plots of age distributions in the selected year
# dec_count=np.zeros(df_countries_year_other.shape[0])
# for i in range(df_countries_year_other.shape[0]):
    
#     dif_list = np.array([j-i for i, j in zip(df_countries_year_other.iloc[i, -22:-2], df_countries_year_other.iloc[i, 9:-1])])
#     dec_count[i] = len(dif_list[dif_list<=0])

# plt.hist(dec_count)
# plt.show()

# for i in range(df_countries_year_other.shape[0]):
    
#     plt.plot(np.cumsum(df_countries_year_other.iloc[i, -22:-1]/df_countries_year_other.iloc[i, -22:-1].sum()), 'b', label="Other")
    
# for i in range(df_countries_year_md.shape[0]):
    
#     plt.plot(np.cumsum(df_countries_year_md.iloc[i, -22:-1]/df_countries_year_md.iloc[i, -22:-1].sum()), 'r--', label = "Monotonic decreasing")
    
# plt.xticks(rotation=45)
# plt.xlabel("Age group")
# plt.show()

# Save 2 sets of age distributions for the selected year, monotonic decreasing (md) and other (i.e. not monotic decreasing)
df_countries_year_md.to_csv(f'{home}data/agedists_countries{select_year}_md.csv') 
df_countries_year_other.to_csv(f'{home}data/agedists_countries{select_year}_other.csv') 