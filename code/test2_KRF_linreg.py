###########################################################################
'''
MODELS OF AGE DISTRIBUTION

Model 1: agents survide according to an age-specific survival probability.
Model 2: agents are subjected to the survival process of model 1 only if they
         are active. The activation probabilities are exogenous.

'''
###########################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings
from joblib import Parallel, delayed

warnings.simplefilter("ignore")
home =  os.getcwd()[:-4] # Set working directory

## DEFINE FUNCTIONS

# def pop_refine(pop_member):
        
#         parm_vectors = analytic_solver(pop_member)
#         activation_rates, survive_probas = parm_vectors
        
#         while any(x > 1 for x in survive_probas):
#             # pop[i,:] = np.concatenate([np.random.rand(dimensions-1)*(1-baseval) + baseval, np.random.rand(1)*baseval + (1-baseval)])
#             pop_member = np.random.rand(dimensions)*(1-baseval) + baseval
#             # pop[i,:] = np.random.rand(dimensions)
#             parm_vectors = analytic_solver(pop_member)
#             activation_rates, survive_probas = parm_vectors
            
            
#         print("fin")
            
#         return pop_member


# def analytic_solver(solution): # Calculate surivival probabilities for given target_dist and p_n
    
#     activation_rates = solution[0:n_groups]
#     survive_probas = np.zeros(n_groups)
    
#     for group in range(n_groups-2):
#         survive_probas[group] = (activation_rates[group+1]*target_dist[group+1])/(activation_rates[group]*target_dist[group])
        
#     survive_probas[(n_groups-1)-1] = (1-solution[n_groups])*((activation_rates[(n_groups-1)]*target_dist[(n_groups-1)])/(activation_rates[(n_groups-1)-1]*target_dist[(n_groups-1)-1]))
    
#     survive_probas[-1]=solution[n_groups]
    
#     parm_vectors = (activation_rates, survive_probas)
    
#     return parm_vectors


def obj_func(solution):
    
    survive_probas = solution[0:n_groups]
    activation_rates = solution[n_groups:n_groups*2]
    
    ages = np.random.randint(0, n_groups, size=pop_size).astype(int)
    ages_history = np.zeros((n_groups, max_itera))
     
    for itera in range(max_itera):
        
        active = activation_rates[ages] < np.random.rand(pop_size)
        
        # active = np.ones(pop_size).astype(bool) # uncomment to use model 1
        trials = np.random.rand(pop_size)
        
        survival_probabilities = survive_probas[ages]
        
        survivals = trials <= survival_probabilities
        deaths = trials > survival_probabilities
  
        ages[active & survivals] += 1
        ages[ages==n_groups] -= 1
        ages[active & deaths] = 0
        
        uages, freqs = np.unique(ages, return_counts=True)
        ages_history[uages, itera] = freqs
     
    ages_final = ages_history[:,-100::].mean(axis=1)
    # ages_dist = np.cumsum(ages_final/ages_final.sum())
    ages_dist_noncum = ages_final/ages_final.sum()
      
    # result = np.mean(np.abs(ages_dist_noncum - target_dist_noncum))

    # return (result, ages_dist_noncum)

    return ages_dist_noncum

## GET DATA

df = pd.read_csv('%sdata/agedists_countries2019_other.csv' % home)

## TEST FUNCTION

pop_size = 10000
max_itera = 1000

numselect=6
target_dist_full = df.iloc[numselect, -22:-1]
target_dist = target_dist_full[target_dist_full>0].copy() #Drop all age classes containing zero individuals (can be absorbed into a neighbouring class)

n_groups = len(target_dist) # Set number of groups to match age classes in data
target_dist_noncum = target_dist/target_dist.sum()
target_dist_cum = np.cumsum(target_dist/target_dist.sum())


# plt.figure(0, figsize=(8,4))

# plt.bar(height=target_dist/target_dist.sum(), x=target_dist.index)
# plt.xticks(rotation=45)
# plt.xlabel("Age group")
# plt.ylabel("P(age==x)")
# plt.title(f"Age distribution: {df.iloc[numselect,3]}")

# plt.tight_layout()
# # plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=300)
# plt.show()


## TEST OPTIMIZATION
popsize = 10
parallel_processes = 64

# Generate random parameter sets
dimensions = 2*n_groups #len(bounds)
baseval = 0 #Minimum value for the initial parameter values
pop_init = np.random.rand(popsize, dimensions)*(1-baseval) + baseval

output = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(obj_func)(sol) for sol in pop_init) # Parallel


plt.figure(0, figsize=(8,4))

plt.scatter(target_dist.index,target_dist_noncum)

for i in range(0,popsize):
    plt.plot(target_dist.index,output[i])
    
plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("P(age==x)")
plt.title(f"Age distribution: {df.iloc[numselect,3]}")

plt.tight_layout()
# plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=300)
plt.show()

















