# -*- coding: utf-8 -*-
"""
Class to compute Pareto Fronts (constrained and unconcstrained) and Hypervolumes for a single multiobjective optimization run

@author: roshan94
"""
from pygmo import hypervolume
import numpy as np

class MOORunStatistics:
    
    # init/constructor method
    def __init__(self, *args):
        self.n_objs = args[0]
        self.objectives = args[1]
        if len(args) > 2:
            self.aggr_constraints = args[2] # aggr_constraints -> sum of absolute constraint violations
        else:
            self.aggr_constraints = []
        self.set_Pareto_idx = []
        self.norm_set_Pareto = []
        self.hypervolume = 0
               
    # method to compute unconstrained Pareto Front indices
    def compute_PF_idx_unconstrained(self):
        pop_size = len(self.objectives)
        obj_num = len(self.objectives[0])

        domination_counter = [0] * pop_size

        for i in range(pop_size):
            for j in range(i+1, pop_size):
                # check each objective for dominance
                dominate = [0] * obj_num
                for k in range(obj_num):
                    if self.objectives[i][k] > self.objectives[j][k]:
                        dominate[k] = 1
                    elif self.objectives[i][k] < self.objectives[j][k]:
                        dominate[k] = -1
                if -1 not in dominate and 1 in dominate:
                    domination_counter[i] += 1
                elif -1 in dominate and 1 not in dominate:
                    domination_counter[j] += 1

        for i in range(len(domination_counter)):
            if domination_counter[i] == 0:
                self.set_Pareto_idx.append(i)
        return self.set_Pareto_idx
    
    # method to compute constrained Pareto Front indices
    def compute_PF_idx_constrained(self, only_fullsat=False):
        if len(self.aggr_constraints) == 0:
            print('Aggregate constraint violations not provided, use compute_PF_unconstrained instead')
        else:
            # Isolate objectives of fully satisfying designs, if required
            if only_fullsat:
                objs = []
                constrs = []
                for i in range(len(self.objectives)):
                    if self.aggr_constraints[i] == 0:
                        objs.append(self.objectives[i])
                        constrs.append(self.aggr_constraints[i])
                objs = np.array(objs)
                constrs = np.array(constrs)
            else:
                objs = self.objectives
                constrs = self.aggr_constraints
                    
            if len(objs) > 0:
                pop_size = len(objs)
                obj_num = len(objs[0])
    
                domination_counter = [0] * pop_size
    
                for i in range(pop_size):
                    for j in range(i+1, pop_size):
                        # First check for aggregate constraint dominance
                        if constrs[i] < constrs[j]:
                            domination_counter[j] += 1
                        elif constrs[i] > constrs[j]:
                            domination_counter[i] += 1
                        else:
                            # For equal constraint satisfaction, check each objective for dominance
                            dominate = [0] * obj_num
                            for k in range(obj_num):
                                if objs[i][k] > objs[j][k]:
                                    dominate[k] = 1
                                elif objs[i][k] < objs[j][k]:
                                    dominate[k] = -1
                            if -1 not in dominate and 1 in dominate:
                                domination_counter[i] += 1
                            elif -1 in dominate and 1 not in dominate:
                                domination_counter[j] += 1
                            
                for i in range(len(domination_counter)):
                    if domination_counter[i] == 0:
                        obj_idx = [j for j in range(len(self.objectives)) if (self.objectives[j,0] == objs[i,0]) and (self.objectives[j,1] == objs[i,1])]
                        self.set_Pareto_idx.append(obj_idx[0])
            return self.set_Pareto_idx
        
    # method to compute hypervolume from the Pareto set (constrained or unconstrained) (can also accomodate a custom population)
    def compute_hv(self, *args):
        if (len(args) == 0):
            population = self.norm_set_Pareto
        else:
            population = args[0]
        
        if len(population) > 0:
            ref = list(np.multiply(np.ones((self.n_objs)),1.1))
            #array_archs = np.zeros((len(population), len(population[0])))
            #for i in range(len(population)):
                #array_archs[i] = population[i]
            hv_object = hypervolume(population)
            self.hypervolume = hv_object.compute(ref)/1.1**self.n_objs
        else:
            self.hypervolume = 0.0
        return self.hypervolume
    
    # method to update Pareto Set objectives (update to normalized objectives before Hypervolume computation)
    def update_norm_PF_objectives(self, norm_PF_objectives):
        self.norm_set_Pareto = norm_PF_objectives