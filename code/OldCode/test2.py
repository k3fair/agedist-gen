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







## TEST FUNCTION

pop_size = 10000
max_itera = 1000
n_groups = 90-18
# survive_probas = 1 - np.random.rand(91)*.1

target_dist = np.concatenate([np.min(np.arange(n_groups/2)*.5)+1+np.arange(n_groups/2)*.5, 
                             np.max(np.arange(n_groups/2)*.5)+1-np.arange(n_groups/2)*.5])
target_dist = np.cumsum(target_dist/target_dist.sum())



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
    ages_dist = np.cumsum(ages_final/ages_final.sum())
    
    return np.max(np.abs(ages_dist - target_dist))





## TEST OPTIMIZATION

best_fitness = 1000
popsize = 64
mut=0.8
crossp=0.7
parallel_processes = 64

bounds = np.array(list(zip(np.zeros(n_groups*2), np.ones(n_groups*2))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
pop =  np.random.rand(popsize, dimensions)*.8 + .2
best_sols = []

step = 0
while True:
    print(step)
    
    fitness = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(obj_func)(sol) for sol in pop) # Parallel
    best_idx = np.argmin(fitness)
    
    if fitness[best_idx] < best_fitness:
        best_sol = pop[best_idx]
        best_fitness = fitness[best_idx]
        best_sols.append(best_sol)
        print(best_fitness)    
        plt.plot(best_sol)
        plt.show()

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
        
    pop = np.array(new_pop)
    step += 1




















