# -*- coding: utf-8 -*-
"""
Optimal Heuristic Weights vs NFE

@author: roshan94
"""
from Utils.dataHandler import DataHandler
#import statistics
import numpy as np
import matplotlib.pyplot as plt
from Utils.mOOCaseStatistics import MOOCaseStatistics

results_dir = 'C:\\SEAK Lab\\Coev results\\'
problem_dir = 'Partition\\' # Truss, Artery ,Assign, Partition, (C1_DTLZ1, Simple DTLZ1_2, UF1 are test problems)

with_IQR = False # whether plot should contain shaded regions for Inter Quartile Range (IQR) or not

int_pop_updated = True
if int_pop_updated:
    int_pop_dir = 'updated int pop\\'
else:
    int_pop_dir = 'same int pop\\'
    
ga_alg = True # Genetic Algorithm (GA) results if True, Differential Evolution (DE) results if False
alg_dir = 'GA\\' 
if not ga_alg: 
    alg_dir = 'DE\\'
    
constr_viol_fitness = True # Constraint Violation based weights fitness if True, feasible HV difference based weights fitness if False
fitness_dir = ''
if problem_dir == 'Truss\\' or problem_dir == 'Artery\\':
    fitness_dir = 'constraint violation fitness\\'
    if not constr_viol_fitness:
        fitness_dir = 'feasible hypervolume fitness\\'
        
weight_of_weights = False # whether an additional weight of weights design decision is used
period_zero_inj = False # whether the zero solution is injected into the population at each 
wow_dir = ''
if weight_of_weights:
    wow_dir = 'WoW - alpha 10e-x\\'
    
pzi_dir = ''
if period_zero_inj:
    pzi_dir = 'PZI - alpha 10e-x\\'

obj_names = ['TrueObjective1','TrueObjective2']
if problem_dir == 'Truss\\' or problem_dir == 'Artery\\':
    heuristic_names = ['P','N','O','I']
    heuristic_names_full = ['PartColl','NodalProp','Orient','Inters']
    colors = ['b','g','r','k']
    #obj_names = ['Normalized Stiffness', 'Normalized Volume Fraction']
    #if problem_dir == 'Artery\\':
        #obj_names = ['Normalized Stiffness', 'Normalized Deviation']
elif problem_dir == 'Assign\\':
    heuristic_names = ['D','O','I','P','M','S','C']
    heuristic_names_full = ['Instrdc','Instrorb','Interinstr','Packeff','Spmass','Instrsyn','Instrcount']
    colors = ['b','g','r','c','m','y','k']
    #obj_names = ['Normalized Science Score', 'Normalized Cost']
else: # problem_dir == 'Partitioning Problem\\' (test problems not considered)
    heuristic_names = ['D','O','I','P','M','S']
    heuristic_names_full = ['Instrdc','Instrorb','Interinstr','Packeff','Spmass','Instrsyn']
    colors = ['b','g','r','c','m','y']

n_runs = 30

alpha = 0.8

case_booleans = [True for i in range(len(heuristic_names))] # All heuristics enforced

optimal_heur_weights = {}
windowed_heur_weights = {}
optimal_weight_mults = {}
windowed_weight_mults = {}
for heur_name in heuristic_names_full:
    optimal_heur_weights[heur_name] = {}
    windowed_heur_weights[heur_name] = {}

heurs_dir = ''
for j in range(len(case_booleans)):
    if case_booleans[j]:
        heurs_dir += heuristic_names[j]
    
heurs_dir += '\\'

# Get NFE values
filepath = results_dir + problem_dir + heurs_dir + int_pop_dir + alg_dir + fitness_dir + wow_dir + pzi_dir + 'run 0\\'
coev_filename = 'coevolutionary_algorithm_heuristic_weights.csv'
coev_full_filename = filepath + coev_filename
data_handler = DataHandler(coev_full_filename)
file_columns = data_handler.read(ignore_nans=False)

nfe_vals = file_columns.get('NFE')

## Store optimal heuristic weights 
dummy_caseStats = MOOCaseStatistics(hv_allcases={}, nfe_array=nfe_vals, case_names=[])
for j in range(len(nfe_vals)):
        
    windowed_weights_nfe = np.zeros((len(heuristic_names_full), n_runs))
    optimal_heur_weights_nfe = np.zeros((len(heuristic_names_full), n_runs))
    optimal_weight_mults_nfe = np.zeros((n_runs))
    windowed_weight_mults_nfe = np.zeros((n_runs))
    
    for i in range(n_runs):
        current_filepath = results_dir + problem_dir + heurs_dir + int_pop_dir + alg_dir + fitness_dir + wow_dir + pzi_dir + 'run ' + str(i) + '\\'
        
        # Read file contents and sort by NFE
        coev_filename = 'coevolutionary_algorithm_heuristic_weights.csv'
        coev_full_filename = current_filepath + coev_filename
    
        data_handler = DataHandler(coev_full_filename)
        file_columns = data_handler.read(ignore_nans=False)
        sorted_file_columns = data_handler.sort_by_nfe()
    
        coev_nfe_vals = sorted_file_columns.get('NFE')
        heur_weights = data_handler.get_heur_weights()
            
        # Determine which heuristic weights are useful (i.e. if fitness is positive, heuristic weights improve constraint satisfaction and/or objective minimization) (NOTE: fitness stored in csv is -fitness since objective minimizer was used)
        fitness = sorted_file_columns.get('Fitness Value 0')
        
        closest_idx = dummy_caseStats.find_closest_index(val=nfe_vals[j], search_list=coev_nfe_vals)
        if -fitness[closest_idx] > 0.0:
            if weight_of_weights:
                optimal_weight_mults_nfe[i] = heur_weights[closest_idx,0]
            for k in range(len(heuristic_names_full)):
                if weight_of_weights:
                    optimal_heur_weights_nfe[k,i] = (10**heur_weights[closest_idx,k+1])*heur_weights[closest_idx,0]
                else:
                    optimal_heur_weights_nfe[k,i] = 10**heur_weights[closest_idx,k]
        else:
            if weight_of_weights:
                optimal_weight_mults_nfe[i] = np.min(heur_weights[:,0])
            for k in range(len(heuristic_names_full)):
                if weight_of_weights:
                    min_idx = np.argmin(heur_weights[k+1])
                    optimal_heur_weights_nfe[k,i] = (10**heur_weights[min_idx,k+1])*heur_weights[min_idx,0]
                else:
                    optimal_heur_weights_nfe[k,i] = 10**np.min(heur_weights[k])
          
        if weight_of_weights:
            if j == 0:
                windowed_weight_mults_nfe[i] = optimal_weight_mults_nfe[i]
            else:
                windowed_weight_mults_nfe[i] = (1 - alpha)*optimal_weight_mults['nfe:'+str(nfe_vals[j-1])][i] + alpha*optimal_weight_mults_nfe[i]
        for k in range(len(heuristic_names_full)):
            if j == 0:
                windowed_weights_nfe[k,i] = optimal_heur_weights_nfe[k,i]
            else:
                windowed_weights_nfe[k,i] = (1 - alpha)*optimal_heur_weights[heuristic_names_full[k]]['nfe:'+str(nfe_vals[j-1])][i] + alpha*optimal_heur_weights_nfe[k,i]
            
    optimal_weight_mults['nfe:'+str(nfe_vals[j])] = optimal_weight_mults_nfe
    windowed_weight_mults['nfe:'+str(nfe_vals[j])] = windowed_weight_mults_nfe
    for k in range(len(heuristic_names_full)):
        optimal_heur_weights[heuristic_names_full[k]]['nfe:'+str(nfe_vals[j])] = optimal_heur_weights_nfe[k,:]
        windowed_heur_weights[heuristic_names_full[k]]['nfe:'+str(nfe_vals[j])] = windowed_weights_nfe[k,:]
        
## Create median and interquartile arrays
windowed_heur_weights_med = {}
if with_IQR:
    windowed_heur_weights_1q = {}
    windowed_heur_weights_3q = {}

for heur_name in heuristic_names_full:
    windowed_heur_weights_med[heur_name] = []
    if with_IQR:
        windowed_heur_weights_1q[heur_name] = []
        windowed_heur_weights_3q[heur_name] = []

for i in range(len(nfe_vals)):
    for heur_name in heuristic_names_full:
        windowed_heur_weights_current_med = np.median(windowed_heur_weights[heur_name]['nfe:'+str(nfe_vals[i])]) 
        if with_IQR:
            windowed_heur_weights_current_1q = np.percentile(windowed_heur_weights[heur_name]['nfe:'+str(nfe_vals[i])], 25) 
            windowed_heur_weights_current_3q = np.percentile(windowed_heur_weights[heur_name]['nfe:'+str(nfe_vals[i])], 75) 
        
        windowed_heur_weights_med[heur_name].append(windowed_heur_weights_current_med)
        if with_IQR:
            windowed_heur_weights_1q[heur_name].append(windowed_heur_weights_current_1q)
            windowed_heur_weights_3q[heur_name].append(windowed_heur_weights_current_3q)
        
## Plot statistics of weights
plt.figure()
heur_counter = 0
for heur_name in heuristic_names_full:
    plt.plot(nfe_vals, windowed_heur_weights_med[heur_name], color=colors[heur_counter], label=heur_name)
    if with_IQR:
        plt.fill_between(nfe_vals, windowed_heur_weights_1q[heur_name], windowed_heur_weights_3q[heur_name], alpha=0.5)
    heur_counter += 1
plt.xlabel('NFE')
plt.ylabel('Heuristic Weight')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best')
#plt.title('Windowed Heuristic Weights vs NFE')

if weight_of_weights:
    plt.figure()
    windowed_weight_mults_med = np.zeros((len(nfe_vals)))
    windowed_weight_mults_1q = np.zeros((len(nfe_vals)))
    windowed_weight_mults_3q = np.zeros((len(nfe_vals)))
    for i in range(len(nfe_vals)):
        windowed_weight_mults_med[i] = np.median(windowed_weight_mults['nfe:'+str(nfe_vals[i])])
        if with_IQR:
            windowed_weight_mults_1q[i] = np.percentile(windowed_weight_mults['nfe:'+str(nfe_vals[i])],25)
            windowed_weight_mults_3q[i] = np.percentile(windowed_weight_mults['nfe:'+str(nfe_vals[i])],75)
    plt.plot(nfe_vals,windowed_weight_mults_med)
    plt.fill_between(nfe_vals, windowed_weight_mults_1q, windowed_weight_mults_3q, alpha=0.5)
    plt.xlabel('NFE')
    plt.ylabel('Heuristic Weight Multiplier')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Windowed Heuristic Weight Multiplier vs NFE')

        
    

    
            

                
        
            
    
    

