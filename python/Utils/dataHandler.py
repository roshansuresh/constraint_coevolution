# -*- coding: utf-8 -*-
"""
Class to read and manipulate data from csv files

@author: roshan94
"""
import csv
import numpy as np

class DataHandler:
    
    # init method/constructor
    def __init__(self, file_loc):
        self.file_loc = file_loc
        self.read_complete = False
        self.line_count = 0
        self.columns = {} # dictionary to store column elements
        
    # Internal method to determine if a string can be converted to a float
    def isfloat(self, val):
        try:
            float(val)
            return True
        except ValueError:
            return False
        
    # read method
    def read(self, ignore_nans):
        with open(self.file_loc, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if self.line_count == 0:
                    for key in row.keys():
                        self.columns[key] = [] # Generate empty lists as items for the column name keys
                        #self.line_count += 1
                if ('NaN' in list(row.values())) and (not ignore_nans):
                    continue
                for k,v in row.items(): # Add each row element to corresponding column list
                    if self.isfloat(v):
                        self.columns[k].append(float(v))
                    elif v == 'TRUE': 
                        self.columns[k].append(True)
                    elif v == 'FALSE':
                        self.columns[k].append(False)
                    else:
                        self.columns[k].append(v)
                self.line_count += 1
            
        self.read_complete = True
        return self.columns
    
    # get line count method
    def get_line_count(self):
        return self.line_count
    
    # get read complete method
    def get_read_complete(self):
        return self.read_complete
    
    # method to sort data by NFE (only for files recording all solutions in optimization run)
    def sort_by_nfe(self):
        if not self.read_complete:
            print('Error: File not read')
        else:
            nfes = self.columns.get('NFE')
            sort_inds = np.argsort(nfes)
            sorted_columns = {}
            for key in self.columns.keys():
                sorted_columns[key] = [self.columns[key][i] for i in sort_inds]
            self.columns = sorted_columns
        return sorted_columns        
    
    # method to get max NFE
    def get_max_nfe(self):
        max_nfe = -1
        if not self.read_complete:
            print('Error: File not read')
        else:
            nfes = self.columns.get('NFE')
            max_nfe = np.max(nfes)
        return max_nfe
    
    # method to obtain the constraint weights as arrays for each NFE (only for the coevolutionary result files)
    def get_constr_weights(self):
        constr_weights = [-1, -1]
        if not self.read_complete:
            print('Error: File not read')
        else:
            constr_weight_cols = [self.columns[col] for col in self.columns.keys() if 'Constraint' in col]
            constr_weights = np.zeros((len(constr_weight_cols[0]),len(constr_weight_cols)))
            for i in range(len(constr_weight_cols[0])): # number of constraint weight combinations
                constr_weight = np.zeros((len(constr_weight_cols)))
                for j in range(len(constr_weight_cols)): # number of heuristics
                    constr_weight[j] = constr_weight_cols[j][i]
                constr_weights[i] = constr_weight
        return constr_weights
    
    # method to obtain objectives as arrays for each NFE (only for the internal MOEA result files)
    def get_objectives(self, obj_names, objs_max, objs_norm_den, objs_norm_num):
        objs = [-1 for k in range(len(obj_names))]
        if not self.read_complete:
            print('Error: File not read')
        else:
            obj_cols = [self.columns[col] for col in self.columns.keys() if col in obj_names]
            objs = np.zeros((len(obj_cols[0]), len(obj_cols)))
            for i in range(len(obj_cols[0])): # number of designs
                obj_current = np.zeros((len(obj_cols)))
                for j in range(len(obj_cols)): # number of objectives
                    obj_val = obj_cols[j][i]*objs_norm_den[j] + objs_norm_num[j] # unnormalize
                    if objs_max[j]: # objective is to be maximized (negative of objective value will be stored)
                        obj_val = -obj_val
                    obj_current[j] = obj_val
                objs[i] = obj_current
        return objs
    
    # method to obtain parameters (constraints or heuristics) as arrays for each NFE (only for the coevolutionary result files)
    def get_parameters(self, parameter_names):
        params = [-1, -1]
        if not self.read_complete:
            print('Error: File not read')
        else:
            param_cols = []
            for col in list(self.columns.keys()):
                if col in parameter_names:
                    param_cols.append(self.columns[col])
            params = np.zeros((len(param_cols[0]), len(param_cols)))
            for i in range(len(param_cols[0])): # number of designs
                param_current = np.zeros((len(param_cols)))
                for j in range(len(param_cols)): # number of constraints
                    param_current[j] = param_cols[j][i]
                params[i] = param_current
        return params
            
    # method to reset fields
    def reset(self):
        self.read_complete = False
        self.line_count = 0
        self.columns = {}
        
        
    
        