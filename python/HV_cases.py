# -*- coding: utf-8 -*-
"""
Compute Hypervolume stats for all runs comparing different cases

@author: roshan94
"""
from Utils.dataHandler import DataHandler
from Utils.normalizationHandler import NormalizationHandler
from Utils.mOORunStatistics import MOORunStatistics
from Utils.mOOCaseStatistics import MOOCaseStatistics
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt

results_dir = 'C:\\SEAK Lab\\Coev Constr results\\'

int_weights = False # whether constraint weights are integers or real values
weight_of_weights = False # whether an additional weight of weights design decision is used
period_zero_inj = False # whether the zero solution is injected into the population at each 

# Plot hypervolume stats vs NFE between two NFE for cases (if needed)
# C1-DTLZ1 -> 3 objs - plot from 12000 to 18000 NFE
hv_plot_range = False
if hv_plot_range:
    nfe_start = 30000   
    nfe_end = 36000

#plot_indiv_pfs = False # SOLELY FOR VALIDATION/DEBUGGING PURPOSES
    
n_runs = 30

case_params = {} # Parameters corresponding to the case (problem, MOEA, coevolutionary or not)
# [moea_choice, problem_choice, cdltz_choice, n_objs, coevolutionary (boolean)]

problem_choice = 1 # 1 -> a C-DTLZ-problem (based on cdtlz_choice), 2 -> Artery problem, 3 -> Equal Stiffness
cdtlz_choice = 1 # 1 -> C1-DTLZ1, 2 -> C1-DTLZ3, 3 -> C2-DTLZ2, 4 -> C3-DTLZ1, 5 -> C3-DTL4 (only for problem_choice = 1)
moea_choice = 1 # 1 -> Epsilon-MOEA, 2 -> MOEA-D, 3 -> IBEA
n_objs = 6 # 3,6 or 12 (only for C-DTLZ problems)

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
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i+1))
                
            constr_names = ['Constraint0']
            
            true_obj_names = obj_names
            if num_objs == 3:
                objs_norm_den = [371, 455, 510]
            elif num_objs == 6:
                objs_norm_den = [221, 201, 296, 418, 499, 464]
            elif num_objs == 12:
                objs_norm_den = [32, 26, 38, 74, 120, 143, 166, 208, 349, 348, 483, 481]
                
            objs_norm_num = list(np.zeros((num_objs)))
            objs_max = [False for j in range(num_objs)]
                
        if c_dtlz_choice == 2:
            problem_dir = 'C1-DTLZ3\\'
            problem_filename = 'c1_dtlz3_'
            
            # Set parameters for DataHandler.get_objectives() method
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i+1))
                
            constr_names = ['Constraint0']
                
            true_obj_names = obj_names
                
            if num_objs == 3:
                objs_norm_den = [1720, 1659, 1729]
            elif num_objs == 6:
                objs_norm_den = [1593, 1761, 1680, 1673, 1741, 1766]
            elif num_objs == 12:
                objs_norm_den = [1413, 1396, 1520, 1685, 1740, 1844, 1813, 1811, 1787, 1834, 1841, 1754]
                
            objs_norm_num = list(np.zeros((num_objs)))
            objs_max = [False for j in range(num_objs)]
                
        elif c_dtlz_choice == 3:
            problem_dir = 'C2-DTLZ2\\'
            problem_filename = 'c2_dtlz2_'
            
            # Set parameters for DataHandler.get_objectives() method            
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i+1))
            true_obj_names = obj_names
            constr_names = ['Constraint0']
                
            if num_objs == 3:
                objs_norm_den = [2.4, 2.5, 2.7]
            elif num_objs == 6:
                objs_norm_den = [2, 2.1, 2.5, 2.4, 2.5, 2.6]
            elif num_objs == 12:
                objs_norm_den = [1.4, 1.2, 1.3, 1.2, 1.7, 1.7, 1.9, 2.2, 2.5, 2.3, 2.6, 2.8]
                
            objs_norm_num = list(np.zeros((num_objs)))
            objs_max = [False for j in range(num_objs)]
                
        elif c_dtlz_choice == 4:
            problem_dir = 'C3-DTLZ1\\'
            problem_filename = 'c3_dtlz1_'
            
            # Set parameters for DataHandler.get_objectives() method            
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i+1))
            true_obj_names = obj_names
            
            for j in range(3):
                constr_names.append('Constraint'+str(j))
                
            if num_objs == 3:
                objs_norm_den = [422, 417, 482]
            elif num_objs == 6:
                objs_norm_den = [403, 388, 454, 452, 482, 473]
            elif num_objs == 12:
                objs_norm_den = [182, 172, 214, 263, 325, 398, 415, 459, 442, 478, 485, 506]
                
            objs_norm_num = list(np.zeros((num_objs)))
            objs_max = [False for j in range(num_objs)]
                
        elif c_dtlz_choice == 5:
            problem_dir = 'C3-DTLZ4\\'
            problem_filename = 'c3_dtlz4_'
            
            # Set parameters for DataHandler.get_objectives() method
            for i in range(num_objs):
                obj_names.append('TrueObjective'+str(i+1))
            true_obj_names = obj_names
            
            for j in range(3):
                constr_names.append('Constraint'+str(j))
                
            if num_objs == 3:
                objs_norm_den = [2.8, 2.6, 2.6]
            elif num_objs == 6:
                objs_norm_den = [2.9, 2.6, 2.6, 2.6, 2.6, 2.6]
            elif num_objs == 12:
                objs_norm_den = [2.9, 2.7, 2.7, 2.7, 2.7, 2.8, 2.8, 2.7, 2.7, 2.8, 2.8, 2.7]
                
            objs_norm_num = list(np.zeros((num_objs)))
            objs_max = [False for j in range(num_objs)]
                
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
        #max_nfe = 10000
    elif cdtlz_choice in [2,3]:
        max_nfe = 12000*n_objs
        #max_nfe = 10000
    else:
        max_nfe = 18000*n_objs
        #max_nfe = 10000
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
                
        
        else: # Coevolutionary MOEA for constraint enforcement - read csv files associated with internal MOEA runs and store collated objectives and constraints
            
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
                
                allzeros = False
                if np.all(constr_weight == -3) and (problem_path == 'Artery\\' or problem_path == 'Equal Stiffness\\'): # for real valued weight decisions, initial set of weights is 0.0
                    constr_weight = np.full((constr_weight.shape[0]), 0.0)
                    allzeros = True
                    
                start_idx = 0
                if weight_of_weights:
                    start_idx = 1
                    
                check_str = prob_filename + str(n_objs) + '_' + moea_filename + 'Constr_Weights-'
                weight_mult = 1
                if weight_of_weights:
                    weight_mult = constr_weight[0]
                for j in range(len(constraint_names)):
                    if int_weights:
                        check_str += 'w' + str(j) + '-' + str(int(constr_weight[j+start_idx])*weight_mult) + ';0' + '_'
                    else:
                        if not allzeros:
                            weight_val = round((10**constr_weight[j+start_idx])*weight_mult, 5) # The values stored in the csv file are raw design decisions, must be converted to actual weights
                        else:
                            weight_val = constr_weight[j+start_idx]
                        if weight_val < 1e-3 and weight_val > 0:
                            check_str += 'w' + str(j) + '-' + np.format_float_scientific(weight_val, exp_digits=1, precision=1, min_digits=1).upper().replace('.',';') + '_' 
                        else:
                            check_str += 'w' + str(j) + '-' + str(weight_val).replace('.',';') + '_' 
                
                if weight_of_weights:
                    if weight_mult < 1e-3 and weight_mult > 0:
                        check_str += 'ww-' + np.format_float_scientific(weight_mult, exp_digits=1, precision=1, min_digits=1).upper().replace('.',';') + '_'
                    else:
                        check_str += 'ww-' + str(round(weight_mult, 5)).replace('.',';') + '_'
                    
                # Search in the current directory to find the results file for the given set of constraint weights and in the current coev_nfe
                file_internal_MOEA = ''
                for root, dirs, files in os.walk(current_filepath):
                    for filename in files:
                        if (check_str in filename) and ('allSolutions' in filename): # check if filename contains the given constraint weights
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
                
                internal_init_nfe = 0
                for j in range(len(internal_file_nfes)):
                    if not coev_nfe == 0:
                        if internal_file_nfes[j] == 0:
                            nfes_current_run.append(coev_nfe*internal_pop_size + internal_file_nfes[j] + internal_init_nfe)
                            internal_init_nfe += 1
                        else:
                            nfes_current_run.append(coev_nfe*internal_pop_size + internal_file_nfes[j])
                    else:
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
n_feas_cases = {}
for case_key in list(case_objs.keys()):
    print('Computing for Case: ' + case_key)
    objs_case = case_objs[case_key]
    nfes_case = case_nfes[case_key]
    constrs_case = case_constrs[case_key]
        
    run_pfs = {}
    n_feas_runs = {}
    for run_key in list(objs_case.keys()):
        print('Computing for Run: ' + run_key)
        objs_run = objs_case[run_key]
        constrs_run = constrs_case[run_key]
        
        nfes_run = nfes_case[run_key]
        
        pfs_run = {}
        n_feas_nfes = {}
        objs_fullsat = []
        for i in range(len(nfe_samples)):
            nfe_sample = nfe_samples[i]
            #if nfe_sample==1300:
                #print('Stop')
            print('Computing for NFE: ' + str(nfe_sample))
            if i != 0:
                closest_prev_nfe_idx = dummy_caseStats.find_closest_index(val=nfe_samples[i-1], search_list=nfes_run)
            else:
                closest_prev_nfe_idx = 0
                current_pf_objs = []
                current_pf_constrs = []
            closest_nfe_idx = dummy_caseStats.find_closest_index(val=nfe_sample, search_list=nfes_run)
            objs_nfe_sample = objs_run[closest_prev_nfe_idx:closest_nfe_idx,:]
            if len(current_pf_objs) != 0:
                objs_nfe_sample = np.append(objs_nfe_sample, current_pf_objs, axis=0)
            objs_current_unique, current_unique_idx = np.unique(objs_nfe_sample, return_index=True, axis=0)
        
            constrs_nfe_sample = constrs_run[closest_prev_nfe_idx:closest_nfe_idx,:]
            if len(current_pf_constrs) != 0:
                constrs_nfe_sample = np.append(constrs_nfe_sample, current_pf_constrs, axis=0)
            # Check and replace indices to account for designs with same objectives but different constraints (replacing with indices for fully feasible designs)
            new_unique_idx = list(current_unique_idx.copy())
            for obj_idx in current_unique_idx:
                eq_obj_idx = [i for i in range(objs_nfe_sample.shape[0]) if (np.array_equal(objs_nfe_sample[i,:], objs_nfe_sample[obj_idx,:]) and i!=obj_idx)]
                if len(eq_obj_idx) > 0:
                    for current_eq_obj_idx in eq_obj_idx:
                        if all(constrs_nfe_sample[current_eq_obj_idx,:] == 0) and any(constrs_nfe_sample[obj_idx,:] != 0):
                            if obj_idx in new_unique_idx:
                                new_unique_idx.remove(obj_idx)
                            new_unique_idx.append(current_eq_obj_idx)
                            break # just need to add one instance of feasible design with same objectives
            
            objs_current_unique = objs_nfe_sample[new_unique_idx,:]
            constr_current_unique = constrs_nfe_sample[new_unique_idx,:]
            
            feas = [x for x in constr_current_unique if np.sum(x) == 0] # find number of fully feasible designs at different NFE
            
            # Add fully feasible objectives to objs_fullsat
            feas_idx = [idx for idx in range(len(constr_current_unique)) if all(constr_current_unique[idx] == 0)]
            for idx_val in feas_idx:
                if not any(np.array_equal(f, objs_current_unique[idx_val,:]) for f in objs_fullsat): # check if design is already present in objs_fullsat
                    objs_fullsat.append(objs_current_unique[idx_val,:])
                
            # Isolate the unique instances of objectives and constraints
            #objs_nfe_sample, unique_idx = np.unique(objs_nfe_sample, return_index=True, axis=0)
            #constrs_nfe_sample = constrs_nfe_sample[unique_idx]
            
            #aggr_constrs_nfe_sample = [np.mean(x) for x in constrs_nfe_sample]
            aggr_constrs_nfe_sample = [np.mean(x) for x in constr_current_unique]
            if len(aggr_constrs_nfe_sample) != 0: 
                #current_mooRunStats = MOORunStatistics(len(objective_names), objs_nfe_sample, aggr_constrs_nfe_sample)
                current_mooRunStats = MOORunStatistics(len(objective_names), objs_current_unique, aggr_constrs_nfe_sample)
                pfs_idx_nfe = current_mooRunStats.compute_PF_idx_constrained(only_fullsat=True)
            else: # would happen if closest_prev_nfe_idx = closest_nfe_idx
                pfs_idx_nfe = []
            n_feas_nfes['nfe:'+str(nfe_sample)] = len(objs_fullsat)
            #if i != 0:
                #n_feas_nfes['nfe:'+str(nfe_sample)] = len(feas) + n_feas_nfes['nfe:'+str(nfe_samples[i-1])]
            #else:
                #n_feas_nfes['nfe:'+str(nfe_sample)] = len(feas) 
            
            #current_pf_objs = objs_nfe_sample[pfs_idx_nfe]
            current_pf_objs = objs_current_unique[pfs_idx_nfe,:]
            current_pf_constrs = constr_current_unique[pfs_idx_nfe,:]
                
            pfs_run['nfe:'+str(nfe_sample)] = current_pf_objs

        run_pfs[run_key] = pfs_run
        n_feas_runs[run_key] = n_feas_nfes
            
    case_pfs[case_key] = run_pfs                
    n_feas_cases[case_key] = n_feas_runs
        
        
#if plot_indiv_pfs: # Plotting individual PFs, SOLELY FOR VALIDATION/DEBUGGING PURPOSES
    #for i in range(n_runs):
        #for nfe_current in nfe_samples:
            #plt.figure()
            #for case_key in list(case_objs.keys()):
                #run_pfs = case_pfs[case_key]
                #run_keys = list(run_pfs.keys())
                #current_run_key = [j for j in range(len(run_keys)) if str(i) in run_keys[j]]
                #current_run_pf = run_pfs[run_keys[current_run_key[0]]]['nfe:'+str(nfe_current)]
                #if current_run_pf.shape[0] > 0:
                    #plt.scatter(current_run_pf[:,0], current_run_pf[:,1], c=plot_colors[case_key], marker=plot_markers[case_key], label=case_key)  
            #plt.xlabel(true_obj_names[0])
            #plt.ylabel(true_obj_names[1])
            #plt.legend(loc='best')
            #plt.title('NFE = ' + str(nfe_current))

## Normalize Pareto Fronts 
print('Normalizing Pareto Front objectives')
norm_handler = NormalizationHandler(objectives=case_pfs, n_objs=len(objective_names))
norm_objs = norm_handler.find_objs_normalization()
norm_handler.normalize_objs()
objs_norm = norm_handler.objectives

## Compute and store hypervolume at sampled nfe for all runs for each case
print('Computing Hypervolumes')
hvs_cases = {}
for case_key in list(objs_norm.keys()):
    pfs_norm_case = objs_norm[case_key]
    hvs_runs = {}
    for run_key in list(pfs_norm_case.keys()):
        pfs_norm_run = pfs_norm_case[run_key]
        hvs_nfes = {}
        for nfe_key in list(pfs_norm_run.keys()):
            pfs_norm_nfe = pfs_norm_run[nfe_key]
            nfe_runStats = MOORunStatistics(len(objective_names), pfs_norm_nfe)
            nfe_runStats.update_norm_PF_objectives(norm_PF_objectives=pfs_norm_nfe)
            hv_nfe = nfe_runStats.compute_hv()
            hvs_nfes[nfe_key] = hv_nfe
        hvs_runs[run_key] = hvs_nfes
    hvs_cases[case_key] = hvs_runs
    
# Compute HV threshold
hv_max = 0
for case_key in list(hvs_cases.keys()):
    hv_runs = hvs_cases[case_key]
    for run_key in list(hv_runs.keys()):
        if np.max(list(hv_runs[run_key].values())) > hv_max:
            hv_max = np.max(list(hv_runs[run_key].values()))

hv_thresh = 0.8*hv_max

## Compute stats (Wilcoxon test, CDF for NFE and median & interquartile ranges for HVs)

#nfe_samples_stats = np.linspace(0, max_nfe, 11) # NFE values at which Wilcoxon tests will be conducted
nfe_samples_stats = nfe_samples

caseStats = MOOCaseStatistics(hv_allcases=hvs_cases, nfe_array=nfe_samples, case_names=list(case_params.keys()))
U_test_cases = caseStats.compute_hypothesis_test_Uvals(nfe_samples=nfe_samples_stats, alternative='two-sided')
nfe_hv_attained = caseStats.compute_nfe_hypervolume_attained(hv_threshold=hv_thresh)
hv_med_cases, hv_1q_cases, hv_3q_cases = caseStats.compute_hypervolume_stats()

## Extract HV stats between two NFE for visualization (if needed)
if hv_plot_range:
    nfe_start_idx = dummy_caseStats.find_closest_index(val=nfe_start, search_list=nfe_samples)
    nfe_end_idx = dummy_caseStats.find_closest_index(val=nfe_end, search_list=nfe_samples)
    nfe_samples_crop = nfe_samples[nfe_start_idx:nfe_end_idx+1]
    
    hv_med_cases_crop = {}
    hv_1q_cases_crop = {}
    hv_3q_cases_crop = {}
    for case_key in list(case_params.keys()):
        hv_med_cases_crop[case_key] = hv_med_cases[case_key][nfe_start_idx:nfe_end_idx+1]
        hv_1q_cases_crop[case_key] = hv_1q_cases[case_key][nfe_start_idx:nfe_end_idx+1]
        hv_3q_cases_crop[case_key] = hv_3q_cases[case_key][nfe_start_idx:nfe_end_idx+1]

## Plotting 
# Plot hypervolume stats vs NFE for cases
fig1 = plt.figure()
for case_key in list(case_params.keys()):
    plt.plot(nfe_samples, hv_med_cases[case_key], label=case_key, color=plot_colors[case_key])
    plt.fill_between(nfe_samples, hv_1q_cases[case_key], hv_3q_cases[case_key], color=plot_colors[case_key], alpha=0.5, edgecolor="none")
plt.xlabel(r'Number of Function Evaluations',fontsize=12)
plt.ylabel(r'Hypervolume',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, borderaxespad=0, prop={"size":12})
plt.savefig("HV_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")

# Plot hypervolume stats vs NFE between two NFE for cases (if needed)
if hv_plot_range:
    fig2 = plt.figure()
    for case_key in list(case_params.keys()):
        plt.plot(nfe_samples_crop, hv_med_cases_crop[case_key], label=case_key, color=plot_colors[case_key])
        plt.fill_between(nfe_samples_crop, hv_1q_cases_crop[case_key], hv_3q_cases_crop[case_key], color=plot_colors[case_key], alpha=0.5, edgecolor="none")
    plt.xlabel(r'Number of Function Evaluations',fontsize=12)
    plt.ylabel(r'Hypervolume',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)    
    plt.savefig("HV_range_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")

# Plot CDF for NFE vs NFE for cases
fig3 = plt.figure()
for case_key in list(case_params.keys()):
    nfe_hv_attained_case = nfe_hv_attained[case_key]
    frac_hv_attained = np.zeros((len(nfe_samples)))
    for i in range(len(nfe_samples)):
        nfe_val = nfe_samples[i]
        idx_hv_attained = [x for x in list(nfe_hv_attained_case.values()) if x <= nfe_val]
        frac_hv_attained[i] = len(idx_hv_attained)/len(list(nfe_hv_attained_case))
    plt.plot(nfe_samples, frac_hv_attained, color=plot_colors[case_key], label=case_key)
plt.xlabel(r'Number of Function Evaluations',fontsize=12)
plt.ylabel(r'Frac. runs achieving 80% max. HV',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, borderaxespad=0, prop={"size":12})
plt.savefig("cdf_nfe_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")

## Plot number of fully feasible designs vs NFE for cases 
fig4 = plt.figure()
n_feas_med_allcases = {}
n_feas_1q_allcases = {}
n_feas_3q_allcases = {}
for case_key in list(n_feas_cases.keys()):
    n_feas_case = n_feas_cases[case_key]
    n_feas_nfes = np.zeros((len(nfe_samples), len(list(n_feas_case.keys()))))
    run_counter = 0
    for run_key in list(n_feas_case.keys()):
        n_feas_run = n_feas_case[run_key]
        nfe_counter = 0
        for nfe_key in list(n_feas_run.keys()):
            n_feas_nfes[nfe_counter,run_counter] = n_feas_run[nfe_key]
            nfe_counter += 1
        run_counter += 1
        
    n_feas_med_case = np.zeros((len(nfe_samples)))
    n_feas_1q_case = np.zeros((len(nfe_samples)))
    n_feas_3q_case = np.zeros((len(nfe_samples)))
    for i in range(len(nfe_samples)):
        n_feas_med_case[i] = statistics.median(n_feas_nfes[i,:])
        n_feas_1q_case[i] = np.percentile(n_feas_nfes[i,:], 25)
        n_feas_3q_case[i] = np.percentile(n_feas_nfes[i,:], 75)
        
    plt.plot(nfe_samples, n_feas_med_case, color=plot_colors[case_key], label=case_key)
    plt.fill_between(nfe_samples, n_feas_1q_case, n_feas_3q_case, color=plot_colors[case_key], alpha=0.5)
plt.xlabel(r'Number of Function Evaluations',fontsize=12)
plt.ylabel(r'# fully feasible solutions',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)    
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=2, borderaxespad=0, prop={"size":12})
plt.savefig("num_feas_" + str(prob_filename) + "_" + str(moea_filename) + "_" + str(n_objs) + ".png")
    
        