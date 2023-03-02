"""
Created on Fri Apr 15 13:34:14 2022

Code containg functions needed to calibrate survival and activation probabilities for, and simulate age distribution with, model 2.

@author: Kathyrn R. Fair, Omar A. Guerrero
"""

###########################################################################
'''
MODELS OF AGE DISTRIBUTION

Model 1: agents survive according to an age-specific survival probability.
Model 2: agents are subjected to the survival process of model 1 only if they
         are active. The activation probabilities are exogenous.

'''
###########################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os, warnings
from joblib import Parallel, delayed

warnings.simplefilter("ignore")
home =  os.getcwd()[:-4] # Set working directory

## DEFINE FUNCTIONS

def pop_refine(pop_member):
        
        parm_vectors = analytic_solver(pop_member)
        activation_rates, survive_probas = parm_vectors
        
        fix_attempt = 0
        
        # Pick survival probabilties
        while any(x > 1 for x in survive_probas):
            # pop[i,:] = np.concatenate([np.random.rand(dimensions-1)*(1-baseval) + baseval, np.random.rand(1)*baseval + (1-baseval)])
            pop_member = np.random.rand(dimensions)*(1-baseval) + baseval
            pop_member[-1] = np.random.rand() #baseval argument is only to pick activation rates to return valid survival probabilties, final entry is p_n (survival probability of oldest age class) and so can just be chosen randomly

            # pop[i,:] = np.random.rand(dimensions)
            parm_vectors = analytic_solver(pop_member)
            activation_rates, survive_probas = parm_vectors
            
            fix_attempt += 1
            
            if fix_attempt == 1e3:
                
                # print("User may want to consider curve fitting approach for their age distribution if this message appears frequently.")
                break # if too many attempts to set the paramters (s.t. survival probabilty values don't exceed 1) have been made move on
        
        # print("done")
        
        return pop_member


def analytic_solver(solution): # Calculate surivival probabilities for given target_dist and p_n
    
    activation_rates = solution[0:n_groups]
    survive_probas = np.zeros(n_groups)
    
    for group in range(n_groups-2):
        survive_probas[group] = (activation_rates[group+1]*target_dist[group+1])/(activation_rates[group]*target_dist[group])
        
    survive_probas[(n_groups-1)-1] = (1-solution[n_groups])*((activation_rates[(n_groups-1)]*target_dist[(n_groups-1)])/(activation_rates[(n_groups-1)-1]*target_dist[(n_groups-1)-1]))
    
    survive_probas[-1]=solution[n_groups]
    
    parm_vectors = (activation_rates, survive_probas)
    
    return parm_vectors


def obj_func(solution): # Run simulation
    
    # survive_probas = solution[0:n_groups]
    # activation_rates = solution[n_groups:n_groups*2]
    parm_vectors = analytic_solver(solution)
    activation_rates, survive_probas = parm_vectors
    
    if all(x <1 for x in survive_probas):
    
        ages = np.random.randint(0, n_groups, size=pop_size).astype(int)
        ages_history = np.zeros((n_groups, max_itera))
        
        for itera in range(max_itera):
            
            active = activation_rates[ages] > np.random.rand(pop_size)
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
        
        result = np.mean(np.abs(ages_dist_noncum - target_dist_noncum))
    
    else:
        result = 1000
        ages_dist_noncum = np.zeros(n_groups)
        # print("bad parametrisation")
        # print(survive_probas-1)
    
    # return np.max(np.abs(ages_dist - target_dist))
    return (result, ages_dist_noncum)

## GET DATA

df = pd.read_csv(f'{home}data/agedists_countries2019_other.csv')

## TEST FUNCTION

pop_size = 10000 # Set number of agents within a simulation
max_itera = 250 # Set total number of timesteps in simulation
# n_groups = 90-18

# survive_probas = 1 - np.random.rand(91)*.1

# target_dist = np.max(np.arange(n_groups)*.5)+1  - np.arange(n_groups)*.5
# target_dist = np.concatenate([np.min(np.arange(n_groups/2)*.5)+1+np.arange(n_groups/2)*.5, 
#                               np.max(np.arange(n_groups/2)*.5)+1-np.arange(n_groups/2)*.5])

# # Artificial target dist
# n_groups = 10
# stepsize=5
# target_dist = np.arange(1,stepsize*(n_groups),stepsize)[::-1]

# target_dist = df.iloc[32, -22:-1] #For now just do the check on the very first entry

# plt.figure(0, figsize=(5,5))

# plt.plot(target_dist/target_dist.sum())
# plt.xticks(rotation=45)
# plt.xlabel("Age group")
# plt.ylabel("P(age==x)")
# plt.title("Age distribution")

# plt.tight_layout()
# plt.show()
        
# n_groups = len(target_dist) # Set number of groups to match age classes in data

# target_dist_cum = np.cumsum(target_dist/target_dist.sum())

numselect=0 # Select index of the age distribution you want to simulate
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
# # plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=300)
# plt.show()


## TEST OPTIMIZATION

best_fitness = 1000
popsize = 100
mut=0.8
crossp=0.7
parallel_processes = -1 #use all available CPUs
algthresh = 7.5e-5 # Set the threshold for stoppping the fitting algorithm

# bounds = np.array(list(zip(np.zeros(n_groups*2), np.ones(n_groups*2))))
bounds = np.array(list(zip(1e-6+np.zeros(n_groups + 1), np.ones(n_groups + 1))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
# pop =  np.random.rand(popsize, dimensions)*.8 + .2
baseval = 0.4 #Minimum value for the initial parameter values
# pop_init = np.concatenate([np.random.rand(popsize, dimensions-1)*(1-baseval) + baseval, np.random.rand(popsize,1)*baseval + (1-baseval)], axis=1)
pop_init = np.random.rand(popsize, dimensions)*(1-baseval) + baseval
pop_init[:,-1] = np.random.rand(popsize) #baseval argument is only to pick activation rates to return valid survival probabilties, final entry is p_n (survival probability of oldest age class) and so can just be chosen randomly

# pop_init =  np.random.rand(popsize, dimensions)
pop = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(pop_refine)(pop_member) for pop_member in pop_init))

best_sols = []

# Plotting parameters
cap = [ "A", "B", "C"]

subcap_x = -0.05
subcap_y = 1.05

step = 0
figcapture=0
while True:
    # print(step)
    
    output = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(obj_func)(sol) for sol in pop) # Parallel
    fitness = [x[0] for x in output]
    best_idx = np.argmin(fitness)
    
    if fitness[best_idx] < best_fitness:
        best_sol = pop[best_idx]
        
        parm_vectors = analytic_solver(best_sol)
        activation_rates, survive_probas = parm_vectors
        
        best_fitness = fitness[best_idx]
        best_sols.append(best_sol)
        print(best_fitness)  
        
        numerical_dist_noncum = obj_func(best_sol)[1]
        
        # numerical_dist_noncum = numerical_dist.copy()
        # numerical_dist_noncum[1:] -= numerical_dist_noncum[:-1].copy()
        
        plt.figure(1, figsize=(6,6))
        
        plt.subplot(211)
        ax1 = sns.lineplot(x=target_dist.index, y =survive_probas, label =  "Survival probability")
        ax1.text(subcap_x, subcap_y, cap[0], transform=ax1.transAxes, size=9, weight='bold')
        sns.lineplot(x=target_dist.index, y =activation_rates, label = "Activation rate")
        plt.xticks(np.arange(n_groups), [])
        plt.legend(title="Parameter", loc = "lower left")
        plt.xlabel("")
        plt.ylabel("Parameter values")
        # plt.title(f"Parameter values: {df.iloc[numselect,3]}")
        
        plt.subplot(212)
        ax2=sns.barplot(x=target_dist.index, y=target_dist_noncum, label = "observed", color = '#1f77b4')
        ax2.text(subcap_x, subcap_y, cap[1], transform=ax2.transAxes, size=9, weight='bold') 
        plt.plot(numerical_dist_noncum, "o--", color="orange", markerfacecolor='none', label = "simulated")
        plt.legend(title=f"Age distribution: {df.iloc[numselect,3]}")
        plt.xticks(rotation=45)
        plt.xlabel(r"Age group ($i$)")
        plt.ylabel(r"P(age group==$i$)")
        # plt.title(f"Age distribution: {df.iloc[numselect,3]}")
        
        # plt.subplot(133)
        # plt.bar(x=target_dist.index, height=target_dist_cum, label = "observed")
        # plt.plot(numerical_dist,  'o-', color="orange", label = "simulated")
        # plt.plot()
        # plt.legend(title="Age distribution")
        # plt.xticks(rotation=45)
        # plt.xlabel("Age group (x)")
        # plt.ylabel("P(age group<=x)")
        # plt.title("Cumulative age distribution")
        
        plt.tight_layout()
        # plt.savefig(f'{home}data/gifimages/agedist_activationmodel_{df.iloc[numselect,3]}_fig{figcapture}.png', bbox_inches="tight", dpi=300)
        plt.savefig(f'{home}data/agedist_{df.iloc[numselect,3]}_model2_activation.png', bbox_inches="tight", dpi=300)
        plt.show()
        
        figcapture += 1
        
    if best_fitness < algthresh:
        
        pd.DataFrame(best_sol).to_csv(f'{home}data/bestparms_{df.iloc[numselect,3]}_model2_activation.csv') 
        print("Desired accuracy has been achieved, ending fitting.")
        break  

    # check = np.array(fitness)
    # len(check[check<1000])
    # print(f"There are {len(check[check<1000])} viable population members.") 
    sorter = np.argsort(fitness)
    survivors = pop[sorter][0:int(len(fitness)/2)].tolist()
    
    new_pop = survivors.copy()
    
    newPop = []
    for j in range(len(survivors)):
        idxs = [idx for idx in range(len(survivors)) if idx != j]
        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(a + mut * (b - c), bounds[:,0], bounds[:,1])
        cross_points = np.random.rand(dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[j])
        trial_denorm = min_b + trial * diff
        new_pop.append(trial_denorm)
    
    pop = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(pop_refine)(pop_member) for pop_member in np.array(new_pop)))
    
    step += 1
    
    if step == 100:
        print("Having gone through 100 iterations of the fitting algorithm without achieving the desired accuracy, you may wish to consider reducing the required accuracy or switching to the curve fitting approach.")




















