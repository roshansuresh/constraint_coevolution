# -*- coding: utf-8 -*-
"""
Comparing Coevolutionary Fitnesses between different cases

@author: roshan94
"""
from Utils.dataHandler import DataHandler
from Utils.mOOCaseStatistics import MOOCaseStatistics
import statistics
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

results_dir = 'C:\\SEAK Lab\\Coev results\\'
problem_dir = 'Assign\\' # Truss, Artery ,Assign, Partition, (C1_DTLZ1, Simple DTLZ1_2, UF1 are test problems)

int_pop_updated = False
if int_pop_updated:
    int_pop_dir = 'updated int pop\\'
else:
    int_pop_dir = 'same int pop\\'
    
int_weights = False # whether heuristic weights are integers or real values
case_booleans = {}

obj_names = ['TrueObjective1','TrueObjective2']
if problem_dir == 'Truss\\' or problem_dir == 'Artery\\':
    heuristic_names = ['P','N','O','I']
    #heuristic_names = ['PartColl','NodalProp','Orient','Inters']
    #obj_names = ['Normalized Stiffness', 'Normalized Volume Fraction']
    #if problem_dir == 'Artery\\':
        #obj_names = ['Normalized Stiffness', 'Normalized Deviation']
elif problem_dir == 'Assign\\':
    heuristic_names = ['Instrdc','Instrorb','Interinstr','Packeff','Spmass','Instrsyn','Instrcount']
    #obj_names = ['Normalized Science Score', 'Normalized Cost']
else: # problem_dir == 'Partitioning Problem\\' (test problems not considered)
    heuristic_names = ['Instrdc','Instrorb','Interinstr','Packeff','Spmass','Instrsyn']

n_runs = 30

case_booleans['All Heurs.'] = [True for i in len(heuristic_names)] # All heuristics enforced
#if problem_dir == 'Truss\\' or problem_dir == 'Artery\\':
    #case_booleans['Prom. Heurs'] = [False, False, True, True] # Only promising heuristics enforced (Orientation and Intersection)
#else: # Assigning or Partitioning problems
    #case_booleans['Prom. Heurs'] = [True, True, True, False, True, True, True] # Only promising heuristics enforced (All except Packing Efficiency)
    
plot_colors = {} # black, yellow for All Heurs. & Prom. Heurs. respectively
plot_colors['All Heurs.'] = '#000000'
#plot_colors['Prom. Heurs.'] = '#E69F00'

# Populate the case fitnesses
case_fitnesses = {}

for key in case_booleans.keys():
    heurs_incorporated = case_booleans[key]
    
    fitness_current_case = {}
    
    heurs_dir = ''
    for j in range(len(heurs_incorporated)):
        if heurs_incorporated[j]:
            heurs_dir += heuristic_names[j]

    if heurs_dir == '': # No heuristics incorporated
        heurs_dir = 'Eps MOEA'
        
    heurs_dir += '\\'
    
    for i in range(n_runs):
        current_filepath = results_dir + problem_dir + heurs_dir + int_pop_dir + 'run ' + str(i) + '\\'
        
        # Read file contents and sort by NFE
        coev_filename = 'coevolutionary_algorithm_heuristic_weights.csv'
        coev_full_filename = current_filepath + coev_filename

        data_handler = DataHandler(coev_full_filename)
        file_columns = data_handler.read(ignore_nans=True)
        sorted_file_columns = data_handler.sort_by_nfe()

        coev_nfe_vals = sorted_file_columns.get('NFE')
        pop_size = 0
        for nfe in coev_nfe_vals:
            if nfe > 0:
                pop_size = nfe
                break

        for j in range(int(pop_size)):
            coev_nfe_vals[j] = float(j)
        max_coev_nfe = np.max(coev_nfe_vals)
        heur_weights = data_handler.get_heur_weights()

        # Finding best fitness values until current NFE
        fitness = sorted_file_columns.get('Fitness Value 0')
        best_fitness = np.zeros((len(fitness)))
        best_fitness[0] = -1*fitness[0]
        for j in range(1,len(coev_nfe_vals)):
            best_fitness[j] = np.max(np.multiply(fitness[:(j+1)], -1))
        
        fitness_current_case['run'+str(i)] = best_fitness
        
    case_fitnesses[key] = fitness_current_case
    
# Compute fitness stats at each NFE for each case
case_fitness_med = {}
case_fitness_1q = {}
case_fitness_3q = {}
for case_key in case_fitnesses.keys():
    fitness_case = case_fitnesses[case_key]
    run_keys = list(fitness_case.keys())
    
    n_nfes = len(coev_nfe_vals)
    
    fitness_med = np.zeros((n_nfes))
    fitness_1q = np.zeros((n_nfes))
    fitness_3q = np.zeros((n_nfes))
    for i in range(n_nfes):
        fitness_vals_nfe = np.zeros((len(run_keys)))
        for j in range(len(run_keys)): # Collect fitness values across all runs at current NFE
            fitnesses_run = fitness_case[run_keys[j]]
            fitness_vals_nfe[j] = fitnesses_run[i]
        # Compute and store fitness stats at current NFE
        fitness_med[i] = statistics.median(fitness_vals_nfe) 
        fitness_1q[i] = np.percentile(fitness_vals_nfe, 25)
        fitness_3q[i] = np.percentile(fitness_vals_nfe, 75)

    case_fitness_med[case_key] = fitness_med
    case_fitness_1q[case_key] = fitness_1q
    case_fitness_3q[case_key] = fitness_3q
    
# Compute Wilcoxon test and t-test statistics and p-values comparing different cases
case_keys = list(case_fitnesses.keys())
cases_ind_array = np.arange(len(case_keys))
case_combs = list(combinations(cases_ind_array, 2))

case_stats_obj = MOOCaseStatistics({}, {}, case_keys) # initialize MOOCaseStatistics object to use internal find_closest_index function

n_samples = 11
linspace_samples_array = np.linspace(0,1,n_samples)
nfe_samples_array = np.floor(np.multiply(linspace_samples_array, coev_nfe_vals[-1])) 

nfe_samples_indices_array = np.zeros(n_samples)
for i in range(len(nfe_samples_array)):
    nfe_samples_indices_array[i] = case_stats_obj.find_closest_index(nfe_samples_array[i], coev_nfe_vals) # indices of nfes to perform Wilcoxon test at

U_allcases_allnfes = {}

for n in range(len(case_combs)):
    #case1_key = case_keys[case_combs[n][0]]
    #case2_key = case_keys[case_combs[n][1]]
    
    U_cases_allnfes = {}
    for nfe_ind in nfe_samples_indices_array:
        
        fits_nfe_cases = {}
        # Extract fitness values for all runs in current case
        for k in range(len(case_combs[n])):
            fit_nfe_case = np.zeros((n_runs))
            run_keys = case_fitnesses[case_keys[case_combs[n][k]]].keys()
            for r in range(len(run_keys)):
                fits_run = case_fitnesses[case_keys[case_combs[n][k]]][run_keys[r]]
                fit_nfe_case[r] = fits_run[nfe_ind]
                
            fits_nfe_cases[case_keys[case_combs[n][k]]] = fit_nfe_case
            
        current_case_keys = list(fits_nfe_cases.keys())
        
        # Perform Wilcoxon Rank Sum Test
        U1, p_val = mannwhitneyu(fits_nfe_cases[current_case_keys[0]], fits_nfe_cases[current_case_keys[1]], alternative='two-sided')
        t_val, p_val_t = ttest_ind(fits_nfe_cases[current_case_keys[0]], fits_nfe_cases[current_case_keys[1]], equal_var=False, alternative='two-sided')
        
        U2 = len(fits_nfe_cases[current_case_keys[0]])*len(fits_nfe_cases[current_case_keys[1]]) - U1
        
        U_test = np.min(np.array([U1, U2]))
        
        U_cases_allnfes['nfe:'+str(nfe_ind)] = [U_test, p_val, p_val_t]
    
    dict_key = case_keys[case_combs[n][0]] + ' and ' + case_keys[case_combs[n][1]]
    U_allcases_allnfes[dict_key] = U_cases_allnfes
    
# Plot fitness stats across all cases
fig = plt.figure()
for case_key in case_fitnesses.keys():
    case_fit_med = case_fitness_med[case_key]
    case_fit_1q = case_fitness_1q[case_key]
    case_fit_3q = case_fitness_3q[case_key]
    
    plt.plot(coev_nfe_vals, case_fit_med, color=plot_colors[case_key], label=case_key)
    plt.fill_between(coev_nfe_vals, case_fit_1q, case_fit_3q, color=plot_colors[case_key], alpha=0.5, edgecolor="none")

plt.xlabel(r'Number of Function Evaluations',fontsize=14)
plt.ylabel(r'Coevolutionary Fitness',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
    
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=3, borderaxespad=0, prop={"size":12})