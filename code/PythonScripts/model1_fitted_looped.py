#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:07:53 2022

Code containing functions needed to create a monotonic decreasing fit to an observed age distribution and simulate an age distribution using survival probabilties calculated based on that fit, using model 1.

@author: Kathyrn R. Fair, Omar A. Guerrero
"""

#Load in required libraries
import pandas as pd
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance

warnings.simplefilter("ignore")

# Set working directory
home =  os.getcwd()[:-18]

## DEFINE FUNCTIONS

def p_n_min_getter(target_dist): # Calculate minimum possible value of p_n
    
    p_n_min = 1 - (target_dist[(n_groups-1)-1]/target_dist[n_groups-1]) 
    
    return p_n_min

def analytic_solver(target_dist, p_n): # Calculate surivival probabilities for given target_dist and p_n
    
    survival_probs_analytical = np.zeros(n_groups)
    
    for group in range(n_groups-2):
        survival_probs_analytical[group] = target_dist[group+1]/target_dist[group]
        
    survival_probs_analytical[(n_groups-1)-1] = (1-p_n)*(target_dist[(n_groups-1)]/target_dist[(n_groups-1)-1])
    
    survival_probs_analytical[-1]=p_n
    
    return survival_probs_analytical
    
def obj_func(survive_probas): # Run simulation
    
    # survive_probas = survive_probas_sol/survive_probas_sol.max()
    ages = np.random.randint(0, n_groups, size=pop_size).astype(int)
    ages_history = np.zeros((n_groups, max_itera))
    
    for itera in range(max_itera):
        
        trials = np.random.rand(pop_size)
        survival_probabilities = survive_probas[ages]
        survivals = trials <= survival_probabilities
        deaths = trials > survival_probabilities
        ages[survivals] += 1
        ages[ages==n_groups] -= 1
        ages[deaths] = 0
        
        uages, freqs = np.unique(ages, return_counts=True)
        ages_history[uages, itera] = freqs
        

        if itera == max_itera-1:
            
            plt.figure(0, figsize=(6,4))
            
            for i in range(n_groups):
                plt.plot(ages_history[i,:], color = palette[i])
            plt.xlabel("Timestep")
            plt.ylabel("# of individuals in age class")
            plt.xlim([0,max_itera-1])
            plt.title(f"Population: {df.iloc[numselect,3]}")
            
            plt.tight_layout()
            plt.savefig(f'timeseries_{df.iloc[numselect,3]}_model1_fitted.png', bbox_inches="tight", dpi=500)
            plt.show()

    # ages_final = np.zeros(n_groups)
    # uages, freqs = np.unique(ages, return_counts=True)
    # ages_final[uages] = freqs
    ages_final = ages_history[:,-100::].mean(axis=1)
    # ages_dist = np.cumsum(ages_final/ages_final.sum())
    ages_dist_noncum = ages_final/ages_final.sum()
    
    return (np.mean(np.abs(ages_dist_noncum - target_dist_noncum)), ages_dist_noncum)

def decay_fcn(x, A, B, C):
    
    y = A*np.ones(len(x))
    y[k:] = A*np.exp(-B*((x[k:] - x[k])**C))
    
    return y


# Read in age distribution data for distributions which are not monotonic decreasing
df_full = pd.read_csv(f'{home}data/required/agedists_countries2019_other.csv')

# Read in list of countries that model 2 was not successful in simulating
algthresh = 1e-4 # Set the threshold for stoppping the fitting algorithm - from model 2, leave as-is
maxsteps = 250 # Maximum number of steps to run of the fitting algorithm - from model 2, leave as-is
df_failed = pd.read_csv(f'{home}data/required/agedists_other_model2failed_thresh{algthresh}_maxsteps{maxsteps}.csv')

# Keep only age distributions for countries where model 2 failed
df = df_full[df_full['Region, subregion, country or area *'].isin(df_failed['0'])]

# df_wassdist = np.zeros(df.shape[0])

# for i in np.arange(0,df.shape[0]):

# target_dist = df.iloc[i, -22:-1] #For now just do the check on the very first entry    
# n_groups = len(target_dist) # Set number of groups to match age classes in data
# target_dist_cum = np.cumsum(target_dist/target_dist.sum())

# failed_attempts = []
wass_storage = []

for i in range(df.shape[0]):

    numselect=i # Select index of the age distribution you want to simulate
    target_dist_full = df.iloc[numselect, -22:-1]
    target_dist = target_dist_full[target_dist_full>0].copy() #Drop all age classes containing zero individuals (can be absorbed into a neighbouring class)
    n_groups = len(target_dist) # Set number of groups to match the number of age classes in the data
    target_dist_noncum = target_dist/target_dist.sum() # Calculate age distribution from counts of individuals in eage age class
    target_dist_cum = np.cumsum(target_dist/target_dist.sum()) # Calcualte cumulative age distribution from counts of individuals in each age class
    
    # plt.figure(0, figsize=(8,4))
    
    # plt.bar(height=target_dist/target_dist.sum(), x=target_dist.index)
    # plt.xticks(rotation=45)
    # plt.xlabel("Age group")
    # plt.ylabel("P(age==x)")
    # plt.title(f"Age distribution: {df.iloc[numselect,3]}")
    
    # plt.tight_layout()
    # plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
    # plt.show()
    
    # plt.figure(0, figsize=(5,5))
    
    # plt.plot(target_dist)
    # plt.xticks(rotation=45)
    # plt.xlabel("Age group")
    # plt.ylabel("P(age==x)")
    # plt.title("Age distribution")
    
    # plt.tight_layout()
    # plt.show()
    
    # Generate array of dummy data matching the length of our target distribution
    xdat = np.arange(0, len(target_dist_cum)) + 1
    
    xdat = np.asarray(xdat)
    ydat = np.asarray(target_dist_cum)
    
    # plt.plot(xdat, ydat, 'o')
    # plt.show()
    
    # # Define the michaelis-menten/monod style equation we're using for fitting
    # def mmfcn(x, A, B, k):
    #     y = A + ((B*x)/(x+k))
    #     return y
    
    # # Fit curve
    # parameters, covariance = curve_fit(mmfcn, xdat, ydat)
    
    # fit_A = parameters[0]
    # fit_B = parameters[1]
    # fit_k = parameters[2]
    # # print(fit_A)
    # # print(fit_B)
    # # print(fit_k)
    
    # fit_y_old = mmfcn(xdat, fit_A, fit_B, fit_k)
    # fit_y_old[fit_y_old>1] = 1 
    
    # # plt.plot(xdat, ydat, 'o', label='data')
    # # plt.plot(xdat, fit_y_old, '-', label='fit')
    # # plt.legend()
    # # plt.show()
    
    # fit_y_noncum = fit_y_old.copy()
    # fit_y_noncum[1:] -= fit_y_noncum[:-1].copy()
    
    # # plt.plot(xdat, target_dist/target_dist.sum() , 'o', label='data')
    # # plt.plot(xdat, fit_y_noncum , '-', label='fit')
    # # plt.legend()
    # # plt.show()
    
    # # print(wasserstein_distance(target_dist_cum, fit_y_old))
    
    best_wass = 100
    best_k = -1
    best_parms = []
    
    for k_val in range(n_groups):
    
        k = k_val
        
        # Fit curve
        ydat = np.asarray(target_dist/target_dist.sum())
        parameters, covariance = curve_fit(decay_fcn, xdat, ydat)
        
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_C = parameters[2]
    
        fit_y = decay_fcn(xdat, fit_A, fit_B, fit_C)
        
        wass_dist = wasserstein_distance(target_dist/target_dist.sum(), fit_y)
        
        plt.figure(0, figsize=(8,4))
        
        plt.bar(x=xdat, height=target_dist_noncum, label='data')
        plt.plot(xdat, fit_y, 's-', color="red", markerfacecolor='none', label='fit')
        plt.xticks(rotation=45, ticks = 1+np.arange(len(target_dist.index)), labels = target_dist.index)
        plt.xlabel(r"Age group ($i$)")
        plt.ylabel(r"P(age group==$i$)")
        plt.legend()
        plt.title(f"Age distribution: {df.iloc[numselect,3]}")
        
        plt.tight_layout()
        # plt.savefig(f'{home}data/agedist_fitted_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
        plt.show()
        
        if (wass_dist < best_wass) & (abs(fit_y.sum()-1) <= 0.01):
            best_wass = wass_dist
            best_k = k
            best_parms = parameters
            
            # print(k)
            # print(wass_dist)
            # print(fit_y.sum())
            # print(parameters)
            
            # print("updated to best fit")
        
    # Set parameters for simulation based on fitting
    fit_A = best_parms[0]
    fit_B = best_parms[1]
    fit_C = best_parms[2]
    k = best_k
    fit_y = decay_fcn(xdat, fit_A, fit_B, fit_C)
    
    # fit_y[fit_y>1] = 1 
    
    # wass_dist = wasserstein_distance(target_dist/target_dist.sum(), fit_y)
    
    # print(f'Wasserstein distance is: {wass_dist}')
    
    # if wass_dist > 0.01:
    
    # plt.figure(1, figsize=(10,5))
    
    # plt.subplot(121)
    # plt.plot(xdat, ydat, 'o', label='data')
    # plt.plot(xdat, fit_y, '-', label='fit')
    # plt.plot(xdat, fit_y_noncum , '--', label='fit (cum-based)')
    
    # plt.subplot(122)
    # plt.plot(xdat, np.cumsum(ydat), 'o', label='data')
    # plt.plot(xdat, np.cumsum(fit_y), '-', label='fit')
    # plt.plot(xdat, fit_y_old, '--', label='fit (cum-based)')
    # plt.legend()
    # plt.show()
    
    # plt.figure(0, figsize=(8,4))
    
    # plt.bar(x=xdat, height=target_dist_noncum, label='data')
    # plt.plot(xdat, fit_y, 's-', color="red", markerfacecolor='none', label='fit')
    # plt.xticks(rotation=45, ticks = 1+np.arange(len(target_dist.index)), labels = target_dist.index)
    # plt.xlabel(r"Age group ($i$)")
    # plt.ylabel(r"P(age group==$i$)")
    # plt.legend()
    # plt.title(f"Age distribution: {df.iloc[numselect,3]}")
    
    # plt.tight_layout()
    # # plt.savefig(f'{home}data/agedist_fitted_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
    # plt.show()
    
    # if abs(fit_y.sum()-1) > 0.01:
        
    #     failed_attempts.append(df.iloc[numselect,3])
    #     print(failed_attempts)
    #     print("Fitted probabilties aren't summing nicely, our fitting procedure may not be suitable for this age distribution.")
    
    # else:
        # df_wassdist[i] = wass_dist
            
        # plt.figure(2, figsize=(10,5))    
        # plt.hist(df_wassdist)
        # plt.show()
        
    ## TEST FUNCTION
    
    pop_size = 10000 # Set number of agents within a simulation
    max_itera = 350 # Set total number of timesteps in simulation
    # n_groups = 90-18
    
    # survive_probas = 1 - np.random.rand(91)*.1
    
    # target_dist = np.max(np.arange(n_groups)*.5)+1  - np.arange(n_groups)*.5
    # target_dist = np.concatenate([np.min(np.arange(n_groups/2)*.5)+1+np.arange(n_groups/2)*.5, 
    #                               np.max(np.arange(n_groups/2)*.5)+1-np.arange(n_groups/2)*.5])
    
    # # Artificial target dist
    # n_groups = 10
    # stepsize=5
    # target_dist = np.arange(1,stepsize*(n_groups),stepsize)[::-1]
    
    # numselect=36
    target_dist_noncum_fitted = fit_y #For now just do the check on the very first entry
    # n_groups = len(target_dist) # Set number of groups to match age classes in data
    # target_dist_noncum = target_dist/target_dist.sum()
    # target_dist_cum = np.cumsum(target_dist/target_dist.sum())
    
    # generate colour palette based on n_groups
    palette = sns.color_palette("flare", n_groups)
    
    # plt.figure(0, figsize=(8,4))
    
    # plt.bar(height=target_dist/target_dist.sum(), x=target_dist.index)
    # plt.xticks(rotation=45)
    # plt.xlabel("Age group")
    # plt.ylabel("P(age==x)")
    # plt.title(f"Age distribution: {df.iloc[numselect,3]}")
    
    # plt.tight_layout()
    # plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
    # plt.show()
    
    
    ## Find minimum p_n value
    p_n_min = p_n_min_getter(target_dist_noncum_fitted)
    
    # print(f"Minimum possible p_n value: {p_n_min_getter(target_dist)}")
    
    if p_n_min <0:
        p_n_min = 0 #If min p_n <0 set =0 since probabilties cannot be <0
    
    # Generate a random p_n value no smaller than our minimum possible p_n value
    p_n = np.random.rand()*(1-p_n_min) + p_n_min
    
    # Generate the corresponding p_i values for i=1,...,n-1
    analytical_survival_probs = analytic_solver(target_dist_noncum_fitted, p_n)
    
    # Run a simulation with this set of surivival probailities
    
    sim_output = obj_func(analytical_survival_probs)
    numerical_dist_noncum = sim_output[1]
    # print(f"Mean absolute error in age-class proportion (noncumuluative) is: {sim_output[0]}")
    
    # Plotting parameters
    cap = [ "A", "B", "C"]
    
    subcap_x = -0.05
    subcap_y = 1.05
    
    label_x = 0.725
    label_y = 0.9
    label_incr = 0.075
    
    # Plot generation
    
    plt.figure(0, figsize=(6,6))
    
    plt.subplot(211)
    ax1=sns.lineplot(x=target_dist.index, y = analytical_survival_probs)
    ax1.text(subcap_x, subcap_y, cap[0], transform=ax1.transAxes, size=9, weight='bold')
    ax1.text(0.1, 0.1, r"$p_{100+}$ = %.2f" % p_n, transform=ax1.transAxes)    
    # plt.legend(title="Solution")
    plt.xticks(np.arange(n_groups), [])
    plt.xlabel("")
    plt.ylabel(r"Survival probability ($p_i$)")
    
    plt.subplot(212)
    ax2=sns.barplot(x=target_dist.index, y=target_dist_noncum, label = "observed", color = '#1f77b4')
    ax2.text(subcap_x, subcap_y, cap[1], transform=ax2.transAxes, size=9, weight='bold') 
    ax2.text(label_x, label_y, "Curve parameters", transform=ax2.transAxes)  
    ax2.text(label_x, label_y - label_incr, "A = %.2f" % fit_A, transform=ax2.transAxes) 
    ax2.text(label_x, label_y - 2*label_incr, "B = %.2f" % fit_B, transform=ax2.transAxes) 
    ax2.text(label_x, label_y - 3*label_incr, "C = %.2f" % fit_C, transform=ax2.transAxes) 
    ax2.text(label_x, label_y - 4*label_incr, "k = %.0f" % k, transform=ax2.transAxes) 
    plt.plot(xdat-1, fit_y, 's-', color="red", markerfacecolor='none', label='fitted curve')
    plt.plot(numerical_dist_noncum, "o--", color="orange", markerfacecolor='none', label = "simulated")
    plt.legend(title=f"Age distribution: {df.iloc[numselect,3]}", loc = 'lower left')
    plt.xticks(rotation=45)
    plt.xlabel(r"Age group ($i$)")
    plt.ylabel(r"P(age group==$i$)")
    # plt.title("Age distribution")
    
    # plt.subplot(133)
    # plt.plot(numerical_dist, label = "simulated")
    # plt.plot(target_dist_cum, label = "observed")
    # plt.plot()
    # plt.legend(title="Age distribution")
    # plt.xticks(rotation=45)
    # plt.xlabel("Age group (x)")
    # plt.ylabel("P(age group<=x)")
    # plt.title("Cumulative age distribution")
    
    plt.tight_layout()
    # plt.savefig(f'{home}data/results/agedist_{df.iloc[numselect,3]}_model1_fitted.png', bbox_inches="tight", dpi=500)
    plt.show()
    
    wass_dist = wasserstein_distance(target_dist_noncum, fit_y)
    wass_storage.append(wass_dist)
    
    # print(f'Wasserstein distance is: {wass_dist}')
        
    print(f'Completion percentage: {100*(i+1)/df.shape[0]}')

# df["wass_val"] = wass_storage # Store wassserstein metric data in original data frame for reference

# #Store lists of which age distributions we weren't able to fit for successfully
# df_failed = pd.DataFrame(failed_attempts)

# df_failed.to_csv(f'{home}data/agedists_other_model1fitted_failed.csv', index=False,encoding='utf-8-sig') 


dpi_val = 100

# Generate histogram of wasserstein metric values
plt.figure(0, figsize=(600/dpi_val, 300/dpi_val), dpi=dpi_val)

sns.histplot(wass_storage, color = '#1f77b4')
plt.ylabel(r"Count")
plt.xlabel("Wasserstein metric")

plt.tight_layout()
plt.savefig(f'{home}data/results/wassdists_model1_fitted.png', bbox_inches="tight", dpi=500)
plt.show()