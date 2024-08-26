# -*- coding: utf-8 -*-
"""
Pareto Fronts at different NFE - heuristic weights for a single run

@author: roshan94
"""
from Utils.dataHandler import DataHandler
from Utils.mOOCaseStatistics import MOOCaseStatistics
from Utils.mOORunStatistics import MOORunStatistics
import numpy as np
import os
import matplotlib.pyplot as plt

results_dir = 'C:\\SEAK Lab\\Coev Constr results\\'

int_weights = False # whether heuristic weights are integers or real values
weight_of_weights = False # whether an additional weight of weights design decision is used
period_zero_inj = False # whether the zero solution is injected into the population at each 

wow_dir = ''
if weight_of_weights:
    wow_dir = 'WoW - alpha 10e-x\\'
    
pzi_dir = ''
if period_zero_inj:
    pzi_dir = 'PZI - alpha 10e-x\\'
    
n_runs = 30

case_params = {} # Parameters corresponding to the case (problem, MOEA, coevolutionary or not)
# [moea_choice, problem_choice, cdltz_choice, n_objs, coevolutionary (boolean)]

problem_choice = 1 # 1 -> a C-DTLZ-problem (based on cdtlz_choice), 2 -> Artery problem, 3 -> Equal Stiffness
cdtlz_choice = 1 # 1 -> C1-DTLZ1, 2 -> C1-DTLZ3, 3 -> C2-DTLZ2, 4 -> C3-DTLZ1, 5 -> C3-DTL4 (only for problem_choice = 1)
moea_choice = 1 # 1 -> Epsilon-MOEA, 2 -> MOEA-D, 3 -> IBEA
n_objs = 3 # 3,6 or 12 (only for C-DTLZ problems)

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

case_params['MOEA'] = [moea_choice, problem_choice, cdtlz_choice, n_objs, False] # Simple MOEA
case_params['CCS'] =  [moea_choice, problem_choice, cdtlz_choice, n_objs, True] # Coevolutionary Constraint Satisfaction

plot_colors = {} # black, blue for MOEA & Coev. Constraint Satisfaction respectively
plot_colors['MOEA'] = '#000000'
plot_colors['CCS'] = '#56B4E9'

plot_markers = {}
plot_markers['MOEA'] = '*'
plot_markers['CCS'] = '+'

## Sample NFEs
nfe_samples = [] 
if problem_choice == 1:
    if cdtlz_choice == 1:
        max_nfe = 6000*n_objs
    elif cdtlz_choice in [2,3]:
        max_nfe = 12000*n_objs
    else:
        max_nfe = 18000*n_objs
else: # Artery or Equal Stiffness
    max_nfe = 13500

# sample nfes are divided into increments of 500 and 1000
n_sample_500 = int(5000/500)
n_sample_1000 = int((max_nfe - 5000)/1000)
for i in range(n_sample_500):
    nfe_samples.append(500*i)
for i in range(n_sample_1000):
    nfe_samples.append((n_sample_500*500) + (1000*i))
nfe_samples.append(max_nfe)

######## POTENTIAL: Script to ascertain pop size from csv file

## Create dictionary of all solution objectives (and constraints if applicable) for all runs for each case (mainly collate internal runs for each coevolutionary run)
case_objs = {}
case_constrs = {}
case_nfes = {}
int_pop_found = False
print('Reading and storing solutions for all cases')

for case_key in list(case_params.keys()):
    run_objs = {}
    run_constrs = {}
    run_nfes = {}
    
    current_case_params = case_params[case_key]
    print('Reading Case: ' + case_key)
    
    moea_path, moea_filename, problem_path, prob_filename, objective_names, constraint_names, coev_path, objs_norm_num, objs_norm_den, objs_max, true_objs_names = obtain_directories_and_names(current_case_params[0], current_case_params[1], current_case_params[2], current_case_params[3], current_case_params[4])
    
    for i in range(n_runs):
        nfes_current_run = []
        current_run_objs = []
        current_run_constrs = []
        print('Reading Run: ' + str(i))
        
        current_filepath = results_dir + problem_path + str(n_objs) + " objectives\\" + moea_path + coev_path + wow_dir + pzi_dir 
        if not current_case_params[4]: # Simple MOEA - just read csv file for corresponding run
        
            run_filename = moea_filename + prob_filename + str(n_objs) + '_' + str(i) + '_allSolutions.csv'
            
            runfile_datahandler = DataHandler(current_filepath + run_filename)
            runfile_cols = runfile_datahandler.read(ignore_nans=False)
            sorted_runfile_cols = runfile_datahandler.sort_by_nfe()
            
            nfes_current_run = sorted_runfile_cols.get('NFE')
            current_run_objs = runfile_datahandler.get_objectives(obj_names=objective_names, objs_max=objs_max, objs_norm_den=objs_norm_den, objs_norm_num=objs_norm_num)
            current_run_constrs = runfile_datahandler.get_parameters(parameter_names=constraint_names)
        
        else: # Coevolutionary MOEA for heuristic enforcement - read csv files associated with internal MOEA runs and store collated objectives and constraints
        
            # Read file contents and sort by NFE
            coev_filename = prob_filename + str(n_objs) + '_' + moea_filename + 'coevolutionary_algorithm_constraint_weights.csv'
            current_filepath += 'run ' + str(i) + '\\'
            coev_full_filename = current_filepath + coev_filename
            
            data_handler = DataHandler(coev_full_filename)
            file_columns = data_handler.read(ignore_nans=True)
            sorted_file_columns = data_handler.sort_by_nfe()
            
            coev_nfe_vals = sorted_file_columns.get('NFE')
            constr_weights = data_handler.get_constr_weights()
            
            #nfes_runfile = []
            #objs_runfile = []
            #constrs_runfile = []
            for n in range(len(coev_nfe_vals)):
                coev_nfe = coev_nfe_vals[n]
                constr_weight = constr_weights[n,:]
                
                start_idx = 0
                if weight_of_weights:
                    start_idx = 1
                
                check_str = ''
                weight_mult = 1
                if weight_of_weights:
                    weight_mult = constr_weight[0]
                for j in range(len(constraint_names)):
                    if int_weights:
                        check_str += 'w' + str(j) + '-' + str(int(constr_weight[j+start_idx])*weight_mult) + ';0' + '_'
                    else:
                        weight_val = round((10**constr_weight[j+start_idx])*weight_mult, 5) # The values stored in the csv file are raw design decisions, must be converted to actual weights
                        if weight_val < 1e-3 and weight_val > 0:
                            check_str += 'w' + str(j) + '-' + np.format_float_scientific(weight_val, exp_digits=1, precision=1, min_digits=1).upper().replace('.',';') + '_' 
                        else:
                            check_str += 'w' + str(j) + '-' + str(weight_val).replace('.',';') + '_' 
                
                if weight_of_weights:
                    if weight_mult < 1e-3 and weight_mult > 0:
                        check_str += 'ww-' + np.format_float_scientific(weight_mult, exp_digits=1, precision=1, min_digits=1).upper().replace('.',';') + '_'
                    else:
                        check_str += 'ww-' + str(round(weight_mult, 5)).replace('.',';') + '_'
                
                # Search in the current directory to find the results file for the given set of heuristic weights and in the current coev_nfe
                file_internal_MOEA = ''
                for root, dirs, files in os.walk(current_filepath):
                    for filename in files:
                        if (check_str in filename) and ('allSolutions' in filename): # check if filename contains the given heuristic weights
                            # Next, check that the NFE value is close to `nfe' (due to limitations in the MOEA Framework implementation the saved NFE is the nearest previous population size multiple)
                            str_nfe_start_ind = filename.find('nfe-')
                            str_nfe_end_ind = filename.rfind('_')
                            nfe_filename = int(filename[str_nfe_start_ind+4:str_nfe_end_ind])
                            if coev_nfe == nfe_filename:
                                file_internal_MOEA = filename
                                break
                    break
                
                # Read the filename, extract and save objectives and constraints
                file_datahandler = DataHandler(current_filepath + file_internal_MOEA)
                
                if not int_pop_found:
                    internal_file_allcols = file_datahandler.read(ignore_nans=True)
                    internal_pop_size = file_datahandler.get_line_count()
                    file_datahandler.reset()
                    int_pop_found = True
                
                internal_file_cols = file_datahandler.read(ignore_nans=False)
                internal_file_sorted_cols = file_datahandler.sort_by_nfe()
                internal_file_nfes = internal_file_sorted_cols.get('NFE')
                internal_objs = file_datahandler.get_objectives(obj_names=objective_names, objs_max=objs_max, objs_norm_den=objs_norm_den, objs_norm_num=objs_norm_num)
                internal_constr = file_datahandler.get_parameters(parameter_names=constraint_names)
            
                for j in range(len(internal_file_nfes)):
                    nfes_current_run.append(coev_nfe*internal_pop_size + internal_file_nfes[j])
                    current_run_objs.append(internal_objs[j,:])
                    current_run_constrs.append(internal_constr[j,:])
                    
        run_nfes['run'+str(i)] = nfes_current_run
        run_objs['run'+str(i)] = np.stack(current_run_objs, axis=0)
        run_constrs['run'+str(i)] = np.stack(current_run_constrs, axis=0)
        
    case_nfes[case_key] = run_nfes
    case_objs[case_key] = run_objs
    case_constrs[case_key] = run_constrs
    
## Create dictionary of Pareto Fronts at sample nfe for all runs for each case
print('Computing and storing Pareto Fronts')
dummy_caseStats = MOOCaseStatistics(hv_allcases={}, nfe_array=nfe_samples, case_names=list(case_params.keys()))
case_pfs = {}
for case_key in list(case_objs.keys()):
    print('Computing for Case: ' + case_key)
    objs_case = case_objs[case_key]
    nfes_case = case_nfes[case_key]
    constrs_case = case_constrs[case_key]
        
    #run_pfs = {}
    nfe_sample_pfs = {}
    for i in range(len(nfe_samples)):
        nfe_sample = nfe_samples[i]
        print('Computing for NFE: ' + str(nfe_sample))
        objs_runs_nfe = []
        constr_runs_nfe = []
        
        for run_key in list(objs_case.keys()):
            objs_run = objs_case[run_key]
            constrs_run = constrs_case[run_key]
            
            nfes_run = nfes_case[run_key]
            
            pfs_run = {}
            objs_fullsat = []
            
            closest_nfe_idx = dummy_caseStats.find_closest_index(val=nfe_sample, search_list=nfes_run)
            objs_nfe_sample = objs_run[:closest_nfe_idx,:]
            #objs_nfe_unique, nfe_unique_idx = np.unique(objs_nfe_sample, return_index=True, axis=0)
            
            if len(objs_runs_nfe) == 0:
                #objs_runs_nfe = objs_nfe_unique
                objs_runs_nfe = objs_nfe_sample
            else:
                #objs_runs_nfe = np.append(objs_runs_nfe, objs_nfe_unique, axis=0)
                objs_runs_nfe = np.append(objs_runs_nfe, objs_nfe_sample, axis=0)
            
            constrs_nfe_sample = constrs_run[:closest_nfe_idx,:]
            #constr_nfe_unique = constrs_nfe_sample[nfe_unique_idx,:]
                    
            if len(constr_runs_nfe) == 0:
                #constr_runs_nfe = constr_nfe_unique
                constr_runs_nfe = constrs_nfe_sample
            else:
                #constr_runs_nfe = np.append(constr_runs_nfe, constr_nfe_unique, axis=0)
                constr_runs_nfe = np.append(constr_runs_nfe, constrs_nfe_sample, axis=0)
        
        objs_runs_nfe_unique, current_unique_idx = np.unique(objs_runs_nfe, return_index=True, axis=0)
        new_unique_idx = list(current_unique_idx.copy())
        for obj_idx in current_unique_idx:
            eq_obj_idx = [i for i in range(objs_runs_nfe.shape[0]) if (np.array_equal(objs_runs_nfe[i,:], objs_runs_nfe[obj_idx,:]) and i!=obj_idx)]
            if len(eq_obj_idx) > 0:
                for current_eq_obj_idx in eq_obj_idx:
                    if all(constr_runs_nfe[current_eq_obj_idx,:] == 0) and any(constr_runs_nfe[obj_idx,:] != 0):
                        if obj_idx in new_unique_idx:
                            new_unique_idx.remove(obj_idx)
                        new_unique_idx.append(current_eq_obj_idx)
                        break # just need to add one instance of feasible design with same objectives
                  
        objs_runs_nfe_unique = objs_runs_nfe[new_unique_idx,:]
        constr_runs_nfe_unique = constr_runs_nfe[new_unique_idx,:]
        
        aggr_constrs_runs_nfe_unique = [np.mean(x) for x in constr_runs_nfe_unique]
        current_mooRunStats = MOORunStatistics(objs_runs_nfe_unique, aggr_constrs_runs_nfe_unique)
        pfs_idx_nfe = current_mooRunStats.compute_PF_idx_constrained(only_fullsat=True)
        
        nfe_pf_objs = objs_runs_nfe_unique[pfs_idx_nfe,:]
        
        nfe_sample_pfs['nfe: ' + str(nfe_sample)] = nfe_pf_objs
    
    case_pfs[case_key] = nfe_sample_pfs

# Plot Pareto Fronts for each case for each sample NFE (Assuming two objectives)
#obj_mult = [-1, 1] 
print('Plotting')
n_figs = len(nfe_samples)
for i in range(n_figs):
    fig = plt.figure()
    for case_key in list(case_objs.keys()):
        nfe_pfs = case_pfs[case_key]
        case_pf_nfe = nfe_pfs['nfe: ' + str(nfe_samples[i])]
                
        #pfs_obj1_case = [x[0] for x in case_pf_nfe]
        #pfs_obj2_case = [x[1] for x in case_pf_nfe]
        
        obj_mult = [1 for j in range(len(objs_max))]
        for k in range(len(obj_mult)): # if objective is negative (since its to be maximized), multiply by -1 for plotting
            if case_pf_nfe[0][k] < 0:
                obj_mult[k] = -1
            
        pfs_obj1 = [np.multiply(x[0], obj_mult[0]) for x in case_pf_nfe]
        pfs_obj2 = [np.multiply(x[1], obj_mult[1]) for x in case_pf_nfe]
        plt.scatter(pfs_obj1, pfs_obj2, c=plot_colors[case_key], marker=plot_markers[case_key], label=case_key)
    
    plt.xlabel(objective_names[0])
    plt.ylabel(objective_names[1])
    plt.legend(loc='best')
    plt.title('NFE = ' + str(nfe_samples[i]))
    
    