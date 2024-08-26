# -*- coding: utf-8 -*-
"""
Plot Fitness Stats across all runs for a single case

@author: roshan94
"""
from Utils.dataHandler import DataHandler
#import statistics
import numpy as np
import matplotlib.pyplot as plt

results_dir = 'C:\\SEAK Lab\\Coev Constr results\\'

weight_of_weights = False # whether an additional weight of weights design decision is used
period_zero_inj = False # whether the zero solution is injected into the population at each 
wow_dir = ''
if weight_of_weights:
    wow_dir = 'WoW - alpha 10e-x\\'
    
pzi_dir = ''
if period_zero_inj:
    pzi_dir = 'PZI - alpha 10e-x\\'

n_runs = 30
coev_pop_size = 6

case_params = {} # Parameters corresponding to the case (problem, MOEA, coevolutionary or not)
# [moea_choice, problem_choice, cdltz_choice, n_objs, coevolutionary (boolean)]

problem_choice = 1 # 1 -> a C-DTLZ-problem (based on cdtlz_choice), 2 -> Artery problem, 3 -> Equal Stiffness
cdtlz_choice = 1 # 1 -> C1-DTLZ1, 2 -> C1-DTLZ3, 3 -> C2-DTLZ2, 4 -> C3-DTLZ1, 5 -> C3-DTL4
moea_choice = 2 # 1 -> Epsilon-MOEA, 2 -> MOEA-D, 3 -> IBEA
n_objs = 3 # 3,6 or 12 (only for C-DTLZ problems)

case_params =  [moea_choice, problem_choice, cdtlz_choice, n_objs, True] # Coevolutionary Constraint Satisfaction

plot_color = '#56B4E9' # blue for Coev. Constraint Satisfaction


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


moea_path, moea_filename, problem_path, prob_filename, objective_names, constraint_names, coev_path, objs_norm_num, objs_norm_den, objs_max, true_objs_names = obtain_directories_and_names(case_params[0], case_params[1], case_params[2], case_params[3], case_params[4])

fitness_case = {}
for i in range(n_runs):
    current_filepath = results_dir + problem_path + str(n_objs) + " objectives\\" + moea_path + coev_path + wow_dir + pzi_dir 
    
    # Read file contents and sort by NFE
    coev_filename = prob_filename + str(n_objs) + '_' + moea_filename + 'coevolutionary_algorithm_constraint_weights.csv'
    coev_full_filename = current_filepath + 'run ' + str(i) + '\\' + coev_filename

    data_handler = DataHandler(coev_full_filename)
    file_columns = data_handler.read(ignore_nans=False)
    sorted_file_columns = data_handler.sort_by_nfe()

    coev_nfe_vals = sorted_file_columns.get('NFE')
    max_coev_nfe = np.max(coev_nfe_vals)
    constr_weights = data_handler.get_constr_weights()

    # Storing fitness values 
    fitness = sorted_file_columns.get('Fitness Value 0')
    fitness_case['run'+str(i)] = np.multiply(fitness,-1) # since Eps MOEA uses a minimizer, -fitness was used as objective for the weights solution 
    
# Compute fitness stats at each NFE
run_keys = fitness_case.keys()

n_nfes = len(coev_nfe_vals)

# Plot fitness stats
n_figs = 6
img_counter = 0
for i in range(n_figs):
    plt.subplot(2,3,i+1)
    for j in range(int(n_runs/n_figs)):
        plt.plot(coev_nfe_vals, fitness_case['run'+str(img_counter)], label='run'+str(img_counter))
        img_counter += 1
    plt.xlabel(r'Number of Function Evaluations',fontsize=12)
    plt.ylabel(r'Coevolutionary Fitness',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
plt.savefig("fitness_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")
    
