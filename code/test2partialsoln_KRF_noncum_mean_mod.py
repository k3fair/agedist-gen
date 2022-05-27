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

def pop_refine(pop_member):
        
        parm_vectors = analytic_solver(pop_member)
        activation_rates, survive_probas = parm_vectors
        
        while any(x > 1 for x in survive_probas):
            # pop[i,:] = np.concatenate([np.random.rand(dimensions-1)*(1-baseval) + baseval, np.random.rand(1)*baseval + (1-baseval)])
            pop_member = np.random.rand(dimensions)*(1-baseval) + baseval
            # pop[i,:] = np.random.rand(dimensions)
            parm_vectors = analytic_solver(pop_member)
            activation_rates, survive_probas = parm_vectors
            
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


def obj_func(solution):
    
    # survive_probas = solution[0:n_groups]
    # activation_rates = solution[n_groups:n_groups*2]
    parm_vectors = analytic_solver(solution)
    activation_rates, survive_probas = parm_vectors
    
    if all(x <1 for x in survive_probas):
    
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
        ages_dist = np.cumsum(ages_final/ages_final.sum())
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

df = pd.read_csv('%sdata/agedists_countries2020_other.csv' % home)

## TEST FUNCTION

pop_size = 10000
max_itera = 1000
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

numselect=6
target_dist_full = df.iloc[numselect, -22:-1] #For now just do the check on the very first entry
target_dist = target_dist_full[target_dist_full>0].copy() #Drop all age classes containing zero individuals (can be absorbed into a neighbouring class)

n_groups = len(target_dist) # Set number of groups to match age classes in data
target_dist_noncum = target_dist/target_dist.sum()
target_dist_cum = np.cumsum(target_dist/target_dist.sum())


plt.figure(0, figsize=(8,4))

plt.bar(height=target_dist/target_dist.sum(), x=target_dist.index)
plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("P(age==x)")
plt.title(f"Age distribution: {df.iloc[numselect,3]}")

plt.tight_layout()
# plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=300)
plt.show()


## TEST OPTIMIZATION

best_fitness = 1000
popsize = 64*4
mut=0.8
crossp=0.7
parallel_processes = 64

# bounds = np.array(list(zip(np.zeros(n_groups*2), np.ones(n_groups*2))))
bounds = np.array(list(zip(1e-6+np.zeros(n_groups + 1), np.ones(n_groups + 1))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
# pop =  np.random.rand(popsize, dimensions)*.8 + .2
baseval = 0.4 #Minimum value for the initial parameter values
# pop_init = np.concatenate([np.random.rand(popsize, dimensions-1)*(1-baseval) + baseval, np.random.rand(popsize,1)*baseval + (1-baseval)], axis=1)
pop_init = np.random.rand(popsize, dimensions)*(1-baseval) + baseval

# pop_init =  np.random.rand(popsize, dimensions)
pop = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(pop_refine)(pop_member) for pop_member in pop_init))

best_sols = []

step = 0
figcapture=0
while True:
    print(step)
    
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
        
        plt.figure(1, figsize=(12,4))
        
        plt.subplot(121)
        plt.plot(survive_probas, label =  "Survival probability")
        plt.plot(activation_rates, label = "Activation rate")
        plt.xticks(np.arange(n_groups), list(target_dist.index), rotation=45)
        plt.legend(title="Parameter", loc = "lower left")
        plt.xlabel("Age group")
        plt.ylabel("Parameter values")
        plt.title(f"Parameter values: {df.iloc[numselect,3]}")
        
        plt.subplot(122)
        plt.bar(x=target_dist.index, height=target_dist_noncum, label = "observed")
        plt.plot(numerical_dist_noncum,  'o-', color="orange", label = "simulated")
        plt.legend(title="Age distribution", loc = "upper right")
        plt.xticks(rotation=45)
        plt.xlabel("Age group (x)")
        plt.ylabel("P(age group==x)")
        plt.title(f"Age distribution: {df.iloc[numselect,3]}")
        
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
        plt.savefig(f'{home}data/gifimages/agedist_activationmodel_{df.iloc[numselect,3]}_fig{figcapture}.png', bbox_inches="tight", dpi=300)
        plt.show()
        
        figcapture += 1

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




















