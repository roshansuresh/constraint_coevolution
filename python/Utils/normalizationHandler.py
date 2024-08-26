# -*- coding: utf-8 -*-
"""
Class to normalize objectives for multiple multiobjective optimization runs

@author: roshan94
"""
import numpy as np

class NormalizationHandler:
    
    def __init__(self, objectives, n_objs):
        self.objectives = objectives # objectives: dict (case -> dict of runs -> dict of nfes) 
        self.n_objs = n_objs # number of objectives
        self.objs_norm = np.zeros((n_objs, 2)) # normalization values for each objective (minimum and maximum)
        self.objs_norm_found = False
        self.n_cases = len(objectives) # number of cases (type of problem such as No heuristic, AOS - All Heuristics etc)
        
    def find_objs_normalization(self):
        
        objs_max_allcases = np.zeros((len(self.objectives), self.n_objs))
        objs_min_allcases = np.zeros((len(self.objectives), self.n_objs))
            
        case_keys = list(self.objectives.keys())
        for i in range(len(case_keys)):
            current_case_objs = self.objectives[case_keys[i]]
            
            objs_max_case = np.zeros((len(current_case_objs), self.n_objs))
            objs_min_case = np.zeros((len(current_case_objs), self.n_objs))
            run_keys = list(current_case_objs.keys())
            
            # Determine and store max and min objectives for each run
            for j in range(len(run_keys)):
                current_run_objs = current_case_objs[run_keys[j]]
                nfe_keys = list(current_run_objs.keys())
                
                objs_max_run = np.zeros((len(current_run_objs), self.n_objs))
                objs_min_run = np.zeros((len(current_run_objs), self.n_objs))
                
                for n in range(len(nfe_keys)):
                    current_nfe_objs = current_run_objs[nfe_keys[n]]
                    
                    if len(current_nfe_objs) > 0:
                        for k in range(self.n_objs):
                            current_nfe_obj = [x[k] for x in current_nfe_objs]
                            objs_max_run[n,k] = np.max(current_nfe_obj)
                            objs_min_run[n,k] = np.min(current_nfe_obj)
                        
                for k in range(self.n_objs):
                    objs_max_case[j,k] = np.max(objs_max_run[:,k])
                    objs_min_case[j,k] = np.min(objs_min_run[:,k])
                    
            # Store in turn the max and min objectives for each case
            for j in range(self.n_objs):
                objs_max_allcases[i,j] = np.max(objs_max_case[:,j])
                objs_min_allcases[i,j] = np.min(objs_min_case[:,j])
                
        # Store overall maximum and minimum objective values for normalization
        for i in range(self.n_objs):
            self.objs_norm[i,0] = np.min(objs_min_allcases[:,i])
            self.objs_norm[i,1] = np.max(objs_max_allcases[:,i])
            
        # Set objs_norm_found to True
        self.objs_norm_found = True
        
        return self.objs_norm
    
    # Method to normalize objectives (after normalization values are found, can also accomodate custom normalization values)
    def normalize_objs(self, *args):
        if ((not self.objs_norm_found) and len(args) == 0):
            print('Normalize values not found yet, run find_objs_normalization first')
        else:
            if (len(args) == 0):
                norm_objs = self.objs_norm
            else:
                norm_objs = args[0]
                
            for case_key in list(self.objectives.keys()):
                current_case_objs = self.objectives[case_key]
                
                for run_key in list(current_case_objs.keys()):
                    current_run_objs = current_case_objs[run_key]
                    
                    for nfe_key in list(current_run_objs.keys()):
                        current_nfe_objs = current_run_objs[nfe_key]
                    
                        for i in range(self.n_objs): # for each objective, do ((obj - min)/(max - min))
                            current_nfe_objs[:,i] = np.divide(np.subtract(current_nfe_objs[:,i], norm_objs[i,0]), (norm_objs[i,1] - norm_objs[i,0]))
                    
                        current_run_objs[nfe_key] = current_nfe_objs
                    
                    current_case_objs[run_key] = current_run_objs
                
                self.objectives[case_key] = current_case_objs
                