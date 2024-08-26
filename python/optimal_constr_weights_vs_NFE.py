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

results_dir = 'C:\\SEAK Lab\\Coev Constr results\\'

n_runs = 30

problem_choice = 1 # 1 -> a C-DTLZ-problem (based on cdtlz_choice), 2 -> Artery problem, 3 -> Equal Stiffness
cdtlz_choice = 1 # 1 -> C1-DTLZ1, 2 -> C1-DTLZ3, 3 -> C2-DTLZ2, 4 -> C3-DTLZ1, 5 -> C3-DTL4 (only for problem_choice = 1)
moea_choice = 2 # 1 -> Epsilon-MOEA, 2 -> MOEA-D, 3 -> IBEA
n_objs = 3 # 3,6 or 12 (only for C-DTLZ problems)

alpha = 0.8

with_IQR = True # whether plot should contain shaded regions for Inter Quartile Range (IQR) or not

weight_of_weights = False # whether an additional weight of weights design decision is used
period_zero_inj = False # whether the zero solution is injected into the population at each 
wow_dir = ''
if weight_of_weights:
    wow_dir = 'WoW - alpha 10e-x\\'
    
pzi_dir = ''
if period_zero_inj:
    pzi_dir = 'PZI - alpha 10e-x\\'

def obtain_directories_and_names(moea_used, prob_choice, c_dtlz_choice, num_objs, coev_used):
    moea_dir = 'Epsilon MOEA\\'
    moea_filename = 'EpsilonMOEA_'
    if moea_used == 2:
        moea_dir = 'MOEA-D\\'
        moea_filename = 'MOEAD_'
    elif moea_used == 3:
        moea_dir = 'IBEA\\'
        moea_filename = 'IBEA_'
        
    problem_dir = '' # Equal Stiffness, Artery or any C-DTLZ problem
    problem_filename = ''
    constr_names = []
    obj_names = []
    if prob_choice == 1:
        if c_dtlz_choice == 1:
            problem_dir = 'C1-DTLZ1\\'
            problem_filename = 'c1_dtlz1_'
            
            # Set parameters for DataHandler.get_objectives() method
            objs_norm_num = [0, 0] 
            objs_max = [False, False] 
            
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i))
                
            constr_names = ['Constraint0']
            
            true_obj_names = obj_names
            if num_objs == 3:
                objs_norm_den = [371, 455, 510]
            elif num_objs == 6:
                objs_norm_den = [221, 201, 296, 418, 499, 464]
            elif num_objs == 12:
                objs_norm_den = [32, 26, 38, 74, 120, 143, 166, 208, 349, 348, 483, 481]
                
        if c_dtlz_choice == 2:
            problem_dir = 'C1-DTLZ3\\'
            problem_filename = 'c1_dtlz3_'
            
            # Set parameters for DataHandler.get_objectives() method
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i))
                
            constr_names = ['Constraint0']
                
            true_obj_names = obj_names
            objs_norm_num = [0, 0] 
            objs_max = [False, False] 
                
            if num_objs == 3:
                objs_norm_den = [1720, 1659, 1729]
            elif num_objs == 6:
                objs_norm_den = [1593, 1761, 1680, 1673, 1741, 1766]
            elif num_objs == 12:
                objs_norm_den = [1413, 1396, 1520, 1685, 1740, 1844, 1813, 1811, 1787, 1834, 1841, 1754]
                
        elif c_dtlz_choice == 3:
            problem_dir = 'C2-DTLZ2\\'
            problem_filename = 'c2_dtlz2_'
            
            # Set parameters for DataHandler.get_objectives() method
            objs_norm_num = [0, 0] 
            objs_max = [False, False] 
            
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i))
            true_obj_names = obj_names
            constr_names = ['Constraint0']
                
            if num_objs == 3:
                objs_norm_den = [2.4, 2.5, 2.7]
            elif num_objs == 6:
                objs_norm_den = [2, 2.1, 2.5, 2.4, 2.5, 2.6]
            elif num_objs == 12:
                objs_norm_den = [1.4, 1.2, 1.3, 1.2, 1.7, 1.7, 1.9, 2.2, 2.5, 2.3, 2.6, 2.8]
                
        elif c_dtlz_choice == 4:
            problem_dir = 'C3-DTLZ1\\'
            problem_filename = 'c3_dtlz1_'
            
            # Set parameters for DataHandler.get_objectives() method
            objs_norm_num = [0, 0] 
            objs_max = [False, False] 
            
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i))
            true_obj_names = obj_names
            
            for j in range(3):
                constr_names.append('Constraint'+str(j))
                
            if num_objs == 3:
                objs_norm_den = [422, 417, 482]
            elif num_objs == 6:
                objs_norm_den = [403, 388, 454, 452, 482, 473]
            elif num_objs == 12:
                objs_norm_den = [182, 172, 214, 263, 325, 398, 415, 459, 442, 478, 485, 506]
                
        elif c_dtlz_choice == 5:
            problem_dir = 'C3-DTLZ4\\'
            problem_filename = 'c3_dtlz4_'
            
            # Set parameters for DataHandler.get_objectives() method
            objs_norm_num = [0, 0] 
            objs_max = [False, False] 
            
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i))
            true_obj_names = obj_names
            
            for j in range(3):
                constr_names.append('Constraint'+str(j))
                
            if num_objs == 3:
                objs_norm_den = [2.8, 2.6, 2.6]
            elif num_objs == 6:
                objs_norm_den = [2.9, 2.6, 2.6, 2.6, 2.6, 2.6]
            elif num_objs == 12:
                objs_norm_den = [2.9, 2.7, 2.7, 2.7, 2.7, 2.8, 2.8, 2.7, 2.7, 2.8, 2.8, 2.7]
                
    elif prob_choice == 2:
        problem_dir = 'Artery\\'
        problem_filename = 'artery_'
        
        # Set parameters for DataHandler.get_objectives() method
        objs_norm_num = [2e5, 0]
        objs_norm_den = [1e6, 1]
        objs_max = [False, False] 
        
        constr_names = ['FeasibilityViolation','ConnectivityViolation']
        true_obj_names = [r'$\frac{C_{11}}{v_f}$',r'deviation']
        for i in range(2):
            obj_names.append('TrueObjective'+str(i))
    
    elif prob_choice == 3:
        problem_dir = 'Equal Stiffness\\'
        problem_filename = 'eqstiffness_'
        
        # Set parameters for DataHandler.get_objectives() method
        objs_norm_num = [0, 0] 
        objs_norm_den = [1.8162e6, 1] # Youngs modulus used to normalize stiffness
        objs_max = [False, False] # To be set to true if negative of any objective is to be used to compute HV, 
        # first objective (stiffness) is to be maximized and second objective (volume fraction/deviation) is to be minimized, however -normalized stiffness is stored in csv so -1 multiplication is not required
        
        constr_names = ['FeasibilityViolation','ConnectivityViolation','StiffnessRatioViolation']
        true_obj_names = [r'$C_{22}$',r'$v_f$']
        for i in range(2):
            obj_names.append('TrueObjective'+str(i))
            
    coev_dir = ''
    if coev_used:
        coev_dir = 'Coevolutionary\\'
            
    return moea_dir, moea_filename, problem_dir, problem_filename, obj_names, constr_names, coev_dir, objs_norm_num, objs_norm_den, objs_max, true_obj_names
    
case_params = [moea_choice, problem_choice, cdtlz_choice, n_objs, True] 

moea_path, moea_filename, problem_path, prob_filename, objective_names, constraint_names, coev_path, objs_norm_num, objs_norm_den, objs_max, true_objs_names = obtain_directories_and_names(case_params[0], case_params[1], case_params[2], case_params[3], case_params[4])

# Generate a random list of colors for each constraint based on the number of constraints
colors = {}
for constr_name in constraint_names:
    colors[constr_name] = np.random.uniform(size=3)

optimal_constr_weights = {}
windowed_constr_weights = {}
optimal_weight_mults = {}
windowed_weight_mults = {}
for constr_name in constraint_names:
    optimal_constr_weights[constr_name] = {}
    windowed_constr_weights[constr_name] = {}

# Get NFE values
filepath = results_dir + problem_path + str(n_objs) + " objectives\\" + moea_path + coev_path + wow_dir + pzi_dir 
coev_filename = prob_filename + str(n_objs) + '_' + moea_filename + 'coevolutionary_algorithm_constraint_weights.csv'
coev_full_filename = filepath + 'run 0\\' + coev_filename
data_handler = DataHandler(coev_full_filename)
file_columns = data_handler.read(ignore_nans=False)

nfe_vals = file_columns.get('NFE')

## Store optimal heuristic weights 
current_filepath = results_dir + problem_path + str(n_objs) + " objectives\\" + moea_path + coev_path + wow_dir + pzi_dir 
dummy_caseStats = MOOCaseStatistics(hv_allcases={}, nfe_array=nfe_vals, case_names=[])
for j in range(len(nfe_vals)):
        
    windowed_weights_nfe = np.zeros((len(constraint_names), n_runs))
    optimal_constr_weights_nfe = np.zeros((len(constraint_names), n_runs))
    optimal_weight_mults_nfe = np.zeros((n_runs))
    windowed_weight_mults_nfe = np.zeros((n_runs))
    
    for i in range(n_runs):        
        
        # Read file contents and sort by NFE
        coev_filename = prob_filename + str(n_objs) + '_' + moea_filename + 'coevolutionary_algorithm_constraint_weights.csv'
        coev_full_filename = current_filepath + 'run ' + str(i) + '\\' + coev_filename
    
        data_handler = DataHandler(coev_full_filename)
        file_columns = data_handler.read(ignore_nans=False)
        sorted_file_columns = data_handler.sort_by_nfe()
    
        coev_nfe_vals = sorted_file_columns.get('NFE')
        constr_weights = data_handler.get_constr_weights()
            
        # Determine which heuristic weights are useful (i.e. if fitness is positive, heuristic weights improve constraint satisfaction and/or objective minimization) (NOTE: fitness stored in csv is -fitness since objective minimizer was used)
        fitness = sorted_file_columns.get('Fitness Value 0')
        
        closest_idx = dummy_caseStats.find_closest_index(val=nfe_vals[j], search_list=coev_nfe_vals)
        if -fitness[closest_idx] > 0.0:
            if weight_of_weights:
                optimal_weight_mults_nfe[i] = constr_weights[closest_idx,0]
            for k in range(len(constraint_names)):
                if weight_of_weights:
                    optimal_constr_weights_nfe[k,i] = (10**constr_weights[closest_idx,k+1])*constr_weights[closest_idx,0]
                else:
                    optimal_constr_weights_nfe[k,i] = 10**constr_weights[closest_idx,k]
        else:
            if weight_of_weights:
                optimal_weight_mults_nfe[i] = np.min(constr_weights[:,0])
            for k in range(len(constraint_names)):
                if weight_of_weights:
                    min_idx = np.argmin(constr_weights[k+1])
                    optimal_constr_weights_nfe[k,i] = (10**constr_weights[min_idx,k+1])*constr_weights[min_idx,0]
                else:
                    optimal_constr_weights_nfe[k,i] = 10**np.min(constr_weights[k])
          
        if weight_of_weights:
            if j == 0:
                windowed_weight_mults_nfe[i] = optimal_weight_mults_nfe[i]
            else:
                windowed_weight_mults_nfe[i] = (1 - alpha)*optimal_weight_mults['nfe:'+str(nfe_vals[j-1])][i] + alpha*optimal_weight_mults_nfe[i]
        for k in range(len(constraint_names)):
            if j == 0:
                windowed_weights_nfe[k,i] = optimal_constr_weights_nfe[k,i]
            else:
                windowed_weights_nfe[k,i] = (1 - alpha)*optimal_constr_weights[constraint_names[k]]['nfe:'+str(nfe_vals[j-1])][i] + alpha*optimal_constr_weights_nfe[k,i]
            
    optimal_weight_mults['nfe:'+str(nfe_vals[j])] = optimal_weight_mults_nfe
    windowed_weight_mults['nfe:'+str(nfe_vals[j])] = windowed_weight_mults_nfe
    for k in range(len(constraint_names)):
        optimal_constr_weights[constraint_names[k]]['nfe:'+str(nfe_vals[j])] = optimal_constr_weights_nfe[k,:]
        windowed_constr_weights[constraint_names[k]]['nfe:'+str(nfe_vals[j])] = windowed_weights_nfe[k,:]
        
## Create median and interquartile arrays
windowed_constr_weights_med = {}
if with_IQR:
    windowed_constr_weights_1q = {}
    windowed_constr_weights_3q = {}

for constr_name in constraint_names:
    windowed_constr_weights_med[constr_name] = []
    if with_IQR:
        windowed_constr_weights_1q[constr_name] = []
        windowed_constr_weights_3q[constr_name] = []

for i in range(len(nfe_vals)):
    for constr_name in constraint_names:
        windowed_constr_weights_current_med = np.median(windowed_constr_weights[constr_name]['nfe:'+str(nfe_vals[i])]) 
        if with_IQR:
            windowed_constr_weights_current_1q = np.percentile(windowed_constr_weights[constr_name]['nfe:'+str(nfe_vals[i])], 25) 
            windowed_constr_weights_current_3q = np.percentile(windowed_constr_weights[constr_name]['nfe:'+str(nfe_vals[i])], 75) 
        
        windowed_constr_weights_med[constr_name].append(windowed_constr_weights_current_med)
        if with_IQR:
            windowed_constr_weights_1q[constr_name].append(windowed_constr_weights_current_1q)
            windowed_constr_weights_3q[constr_name].append(windowed_constr_weights_current_3q)
        
## Plot statistics of weights
plt.figure()
#constr_counter = 0
for constr_name in constraint_names:
    plt.plot(nfe_vals, windowed_constr_weights_med[constr_name], color=colors[constr_name], label=constr_name)
    if with_IQR:
        plt.fill_between(nfe_vals, windowed_constr_weights_1q[constr_name], windowed_constr_weights_3q[constr_name], alpha=0.5)
plt.xlabel('NFE')
plt.ylabel('Constraint Weight')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best')
#plt.title('Windowed Heuristic Weights vs NFE')
plt.savefig("weights_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")

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
    plt.savefig("wow_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")

        
    

    
            

                
        
            
    
    

