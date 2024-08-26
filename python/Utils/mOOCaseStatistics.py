# -*- coding: utf-8 -*-
"""
Class to compute comparison statistics for multiple cases of multiobjective optimization runs

@author: roshan94
"""
import numpy as np
import statistics
from itertools import combinations
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

class MOOCaseStatistics:
    
    def __init__(self, hv_allcases, nfe_array, case_names):
        self.hv_allcases = hv_allcases # dictionary of hypervolume values for all runs in each case (length = number of cases)
        self.nfe_array = nfe_array # Array of NFE values corresponding to the HV values 
        self.case_names = case_names # case names to store hypothesis testing results
        self.n_cases = len(hv_allcases)
        
    # Internal method 
    def find_closest_index(self, val, search_list):
        val_diff = np.array(search_list) - val
        closest_val = search_list[np.argmin(np.abs(val_diff))]
        idx = len(search_list) - search_list[::-1].index(closest_val) - 1
        return idx
        
    # Method to compute hypothesis test statistics and p-values (Wilcoxon Rank Sum and t-tests)
    def compute_hypothesis_test_Uvals(self, nfe_samples, alternative): 
        # nfe_samples: array of NFE values for hypothesis testing 
        # alternative: alternative testing hypothesis -> 'more', 'less' or 'two-sided'
        
        #n_samples = len(nfe_samples)
        
        nfe_samples_indices_array = np.zeros(len(nfe_samples))
        for i in range(len(nfe_samples)):
            nfe_samples_indices_array[i] = self.find_closest_index(nfe_samples[i], self.nfe_array)
            
        hv_samples_allcases_allruns = {}
        case_keys = list(self.hv_allcases.keys())
        for case_key in case_keys:
            hv_dict_allruns_currentcase = self.hv_allcases[case_key]
            hv_samples_allruns = {}
            for k in range(len(nfe_samples_indices_array)):
                hv_samples_nfe_allruns = np.zeros(len(hv_dict_allruns_currentcase))
                run_keys = list(hv_dict_allruns_currentcase.keys())
                for m in range(len(run_keys)):
                    hv_run = hv_dict_allruns_currentcase[run_keys[m]]
                    hv_samples_nfe_allruns[m] = list(hv_run.values())[int(nfe_samples_indices_array[k])]
                hv_samples_allruns['nfe:'+str(int(nfe_samples[k]))] = hv_samples_nfe_allruns
            hv_samples_allcases_allruns[case_key] = hv_samples_allruns
            
        cases_inds_array = np.arange(self.n_cases)
        case_combinations = list(combinations(cases_inds_array,2))
        
        U_test_cases = {}
        
        for n in range(len(case_combinations)):
            case_string_x = case_keys[case_combinations[n][0]]
            case_string_y = case_keys[case_combinations[n][1]]
            
            hv_allruns_casex = hv_samples_allcases_allruns[case_string_x]
            hv_allruns_casey = hv_samples_allcases_allruns[case_string_y]
            
            U_test_cases_allnfes = {}
            for nfe_key in list(hv_allruns_casex.keys()):
                hv_samples_nfe_casex = hv_allruns_casex[nfe_key]
                hv_samples_nfe_casey = hv_allruns_casey[nfe_key]
                
                U1, p_val = mannwhitneyu(hv_samples_nfe_casex, hv_samples_nfe_casey, alternative=alternative)
                t_val, p_val_t = ttest_ind(hv_samples_nfe_casex, hv_samples_nfe_casey, equal_var=False, alternative=alternative)
                
                U2 = len(hv_samples_nfe_casex)*len(hv_samples_nfe_casey) - U1
                
                U_test = np.min(np.array([U1, U2]))
                
                U_test_cases_allnfes[nfe_key] = [U_test, p_val, p_val_t]
            
            dict_key = case_string_x + ' and ' + case_string_y
            U_test_cases[dict_key] = U_test_cases_allnfes
                
        return U_test_cases

    # Method to compute array of NFE values for reaching threshold hypervolume for different runs of a all cases (not particular case)
    def compute_nfe_hypervolume_attained(self, hv_threshold):
        #hv_threshold = 0.75 # Threshold HV value to reach, user parameter
        
        nfe_hv_attained_cases = {}
        for case_key in list(self.hv_allcases.keys()):
            #case_key = case_keys[i]
            hv_dict_case = self.hv_allcases[case_key]
            #run_keys = list(hv_dict_case.keys())
            
            nfe_hv_attained_runs = {}
            for run_key in list(hv_dict_case.keys()):
                #run_key = run_keys[j]
                hv_dict_run = hv_dict_case[run_key]
                nfe_keys = list(hv_dict_run.keys())
                hv_vals_run = list(hv_dict_run.values())
                
                nfe_vals = [int(x[4:]) for x in nfe_keys] # extract NFE values from dict keys
                
                nfe_hv_attained_run = nfe_vals[-1] + 100
                for i in range(len(hv_vals_run)):
                    if (hv_vals_run[i] >= hv_threshold):
                        nfe_hv_attained_run = nfe_vals[i]
                        break
                    
                nfe_hv_attained_runs[run_key] = nfe_hv_attained_run
            
            nfe_hv_attained_cases[case_key] = nfe_hv_attained_runs
            
        return nfe_hv_attained_cases

    # Method to compute hypervolume stats for all runs and all cases 
    def compute_hypervolume_stats(self):
        case_keys = list(self.hv_allcases.keys())
        n_datapoints = len(self.nfe_array)
        hv_median_allcases = {}#np.zeros(n_datapoints)
        hv_1q_allcases = {}#np.zeros(n_datapoints)
        hv_3q_allcases = {}#np.zeros(n_datapoints)
        for case_key in case_keys:
            hv_case = self.hv_allcases[case_key]
            hv_median_case = np.zeros(n_datapoints)
            hv_1q_case = np.zeros(n_datapoints)
            hv_3q_case = np.zeros(n_datapoints)
            for i in range(n_datapoints):
                hv_vals = []
                for run_key in hv_case.keys():
                    hv_run = hv_case[run_key]
                    hv_current_array = list(hv_run.values())
                    hv_vals.append(hv_current_array[i])
                hv_median_case[i] = statistics.median(hv_vals)
                hv_1q_case[i] = np.percentile(hv_vals, 25)
                hv_3q_case[i] = np.percentile(hv_vals, 75)
            hv_median_allcases[case_key] = hv_median_case
            hv_1q_allcases[case_key] = hv_1q_case
            hv_3q_allcases[case_key] = hv_3q_case
            
        return hv_median_allcases, hv_1q_allcases, hv_3q_allcases
            