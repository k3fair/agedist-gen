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

def pop_refine(pop_member): #Updates any initial parameter guesses that would violate requirement that probabilities not be >1
        
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


def obj_func_solver(solution):
    
    # survive_probas = solution[0:n_groups]
    # activation_rates = solution[n_groups:n_groups*2]
    parm_vectors = analytic_solver(solution)
    activation_rates, survive_probas = parm_vectors
    
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
    ages_dist_noncum = ages_final/ages_final.sum()

    return ages_dist_noncum, activation_rates, survive_probas

def obj_func_simpler(activation_rates, survive_probas):
    
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
    ages_dist_noncum = ages_final/ages_final.sum()

    return ages_dist_noncum

def unpacker(output,entry): #To unpack results from simulations
    
    for run in range(len(output)):
        run_result = output[run][entry]
        
        if run == 0:
            result = run_result
        else:
            result = np.vstack((result, run_result))

    return result

## GET DATA

df = pd.read_csv('%sdata/agedists_countries2019_md.csv' % home)

## TEST FUNCTION

pop_size = 1000
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

numselect=37 #6
target_dist_full = df.iloc[numselect, -22:-1] #For now just do the check on the very first entry
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

popsize = 5000
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

output = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(obj_func_solver)(sol) for sol in pop) # Parallel

ages_dist_noncum_result = unpacker(output,0)
activation_rates_result = unpacker(output,1)
survive_probas_result = unpacker(output,2)

# plt.figure(0, figsize=(8,4))

# for i in range(popsize):
#     plt.plot(target_dist.index,ages_dist_noncum_result[i])

# plt.scatter(target_dist.index,target_dist_noncum)
    
# plt.xticks(rotation=45)
# plt.xlabel("Age group")
# plt.ylabel("P(age==x)")
# plt.title(f"Age distribution: {df.iloc[numselect,3]}")

# plt.tight_layout()
# # plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=300)
# plt.show()

### Regression step

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNetCV

#Create arrays to store predicted values
activation_rates_fitted = np.zeros(n_groups)
survive_probas_fitted = np.zeros(n_groups)

# First regressions for the activation rates
for k in range(n_groups):
    # load the dataset
    data = ages_dist_noncum_result
    parm = activation_rates_result[:,k]
    X, y = data, parm
    # # define model evaluation method
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    # ratios = np.arange(0, 1, 0.01)
    # alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    # model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
    model = ElasticNetCV(n_jobs=-1)
    # fit model
    model.fit(X, y)
    # # summarize chosen configuration
    # print('alpha: %f' % model.alpha_)
    # print('l1_ratio_: %f' % model.l1_ratio_)
    # # summarise results
    # print('intercept_: %f' % model.intercept_)
    # print('coefs:', model.coef_)
    # # show predicted value
    # print('prediction: %f' % model.predict(np.array(target_dist_noncum).reshape(1,-1)))
    
    activation_rates_fitted[k] = model.predict(np.array(target_dist_noncum).reshape(1,-1))

# Then regressions for the survival probabilities
for k in range(n_groups):
    # load the dataset
    data = ages_dist_noncum_result
    parm = survive_probas_result[:,k]
    X, y = data, parm
    # # define model evaluation method
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    # ratios = np.arange(0, 1, 0.01)
    # alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    # model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
    model = ElasticNetCV(n_jobs=-1)
    # fit model
    model.fit(X, y)
    # # summarize chosen configuration
    # print('alpha: %f' % model.alpha_)
    # print('l1_ratio_: %f' % model.l1_ratio_)
    # # summarise results
    # print('intercept_: %f' % model.intercept_)
    # print('coefs:', model.coef_)
    # # show predicted value
    # print('prediction: %f' % model.predict(np.array(target_dist_noncum).reshape(1,-1)))
    
    survive_probas_fitted[k] = model.predict(np.array(target_dist_noncum).reshape(1,-1))
    

plt.figure(1, figsize=(8,4))

# plt.plot(target_dist.index,obj_func_simpler(activation_rates_fitted, survive_probas_fitted), label = "predicted")
# plt.scatter(target_dist.index,target_dist_noncum, label = "observed")

plt.bar(x=target_dist.index, height=target_dist_noncum, label = "observed")
plt.plot(target_dist.index,obj_func_simpler(activation_rates_fitted, survive_probas_fitted),  'o-', color="orange", label = "simulated")
plt.legend(title="Age distribution", loc = "upper right")

plt.xticks(rotation=45)
plt.xlabel("Age group")
plt.ylabel("P(age==x)")
plt.title(f"Age distribution: {df.iloc[numselect,3]}")

plt.tight_layout()
# plt.savefig(f'agedist_empirical_{df.iloc[numselect,3]}.png', bbox_inches="tight", dpi=300)
plt.show()