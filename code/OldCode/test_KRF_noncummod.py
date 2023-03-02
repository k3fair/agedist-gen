import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, warnings
from joblib import Parallel, delayed

warnings.simplefilter("ignore")
home =  os.getcwd()[:-4] # Set working directory

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
    
def obj_func(survive_probas):
    
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

    # ages_final = np.zeros(n_groups)
    # uages, freqs = np.unique(ages, return_counts=True)
    # ages_final[uages] = freqs
    ages_final = ages_history[:,-100::].mean(axis=1)
    ages_dist = np.cumsum(ages_final/ages_final.sum())
    ages_dist_noncum = ages_final/ages_final.sum()
    
    return (np.max(np.abs(ages_dist_noncum - target_dist_noncum)), ages_dist)

# def obj_func(survive_probas):
    
#     # survive_probas = survive_probas_sol/survive_probas_sol.max()
#     ages = np.random.randint(0, n_groups, size=pop_size).astype(int)
#     ages_history = np.zeros((n_groups, max_itera))
    
#     for itera in range(max_itera):
        
#         trials = np.random.rand(pop_size)
#         survival_probabilities = survive_probas[ages]
#         survivals = trials <= survival_probabilities
#         deaths = trials > survival_probabilities
#         ages[survivals] += 1
#         ages[ages==n_groups] -= 1
#         ages[deaths] = 0
        
#         uages, freqs = np.unique(ages, return_counts=True)
#         ages_history[uages, itera] = freqs

#     # ages_final = np.zeros(n_groups)
#     # uages, freqs = np.unique(ages, return_counts=True)
#     # ages_final[uages] = freqs
#     ages_final = ages_history[:,-100::].mean(axis=1)
#     ages_dist_probs = ages_final/ages_final.sum()
    
#     return (np.max(np.abs(ages_dist_probs - target_dist_probs)), ages_dist_probs)


# ## TEST CODE
# pop_size = 10000
# max_itera = 1000
# survive_probas = 1 - np.random.rand(91)*.1

# ages = np.random.randint(18, 91, size=pop_size).astype(int)
# ages_history = np.zeros((91, max_itera))

# for itera in range(max_itera):
    
#     trials = np.random.rand(pop_size)
#     survival_probabilities = survive_probas[ages]
#     survivals = trials <= survival_probabilities
#     deaths = trials > survival_probabilities
#     ages[survivals] += 1
#     ages[ages>90] = 90
#     ages[deaths] = 18
    
#     uages, freqs = np.unique(ages, return_counts=True)
#     ages_history[uages, itera] = freqs

# for serie in ages_history:
#     plt.plot(serie)

## GET DATA

df = pd.read_csv('%sdata/agedists_countries2020_md.csv' % home)


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

numselect=37
target_dist = df.iloc[numselect, -22:-1] #For now just do the check on the very first entry
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
plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=500)
plt.show()


## p_n check
print(f"Minimum possible p_n value: {p_n_min_getter(target_dist)}")

## TEST OPTIMIZATION

best_fitness = 1000
popsize = 64
mut=0.8
crossp=0.7
parallel_processes = 64

bounds = np.array(list(zip(np.zeros(n_groups), np.ones(n_groups))))
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
dimensions = len(bounds)
pop =  np.random.rand(popsize, dimensions)*.8 + .2
best_sols = []

step = 0
while True:
    print(step)
    
    output = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(obj_func)(sol) for sol in pop) # Parallel
    fitness = [x[0] for x in output]
    best_idx = np.argmin(fitness)
    
    if fitness[best_idx] < best_fitness:
        best_sol = pop[best_idx]
        best_fitness = fitness[best_idx]
        best_sols.append(best_sol)
        print(best_fitness) 
        
        numerical_dist = obj_func(best_sol)[1]
        
        numerical_dist_noncum = numerical_dist.copy()
        numerical_dist_noncum[1:] -= numerical_dist_noncum[:-1].copy()
        
        plt.figure(0, figsize=(12,4))
        
        plt.subplot(131)
        plt.plot(best_sol, label = "numerical")
        plt.plot(analytic_solver(target_dist, best_sol[-1]), label = "analytical")
        plt.legend(title="Solution")
        plt.xticks(np.arange(n_groups), list(target_dist.index), rotation=45)
        plt.xlabel("Age group")
        plt.ylabel("Survival probability")
        
        plt.subplot(132)
        plt.plot(numerical_dist_noncum, label = "simulated")
        plt.plot(target_dist_noncum, label = "observed")
        plt.plot()
        plt.legend(title="Age distribution")
        plt.xticks(rotation=45)
        plt.xlabel("Age group (x)")
        plt.ylabel("P(age group==x)")
        plt.title("Age distribution")
        
        plt.subplot(133)
        plt.plot(numerical_dist, label = "simulated")
        plt.plot(target_dist_cum, label = "observed")
        plt.plot()
        plt.legend(title="Age distribution")
        plt.xticks(rotation=45)
        plt.xlabel("Age group (x)")
        plt.ylabel("P(age group<=x)")
        plt.title("Cumulative age distribution")
        
        plt.tight_layout()
        plt.show()


    sorter = np.argsort(fitness)
    survivors = pop[sorter][0:int(len(fitness)/2)].tolist()
    new_pop = survivors.copy()
    
    newPop = []
    for j in range(len(survivors)):
        idxs = [idx for idx in range(len(survivors)) if idx != j]
        a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(a + mut * (b - c), 10e-12, 1)
        cross_points = np.random.rand(dimensions) < crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimensions)] = True
        trial = np.where(cross_points, mutant, pop[j])
        trial_denorm = min_b + trial * diff
        new_pop.append(trial_denorm)
        
    pop = np.array(new_pop)
    step += 1





















