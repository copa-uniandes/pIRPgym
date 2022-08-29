################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time; from termcolor import colored
from numpy.random import seed, normal, lognormal, randint, uniform, random

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

### Gym & OR-Gym
import utils


class instance_generator():

    def __init__(self, env, rd_seed):
        
        seed(rd_seed)
        
        ### Importing instance parameters ###
        self.M = env.M; self.Suppliers = env.Suppliers    # Suppliers
        self.V = env.V
        self.K = env.K; self.Products = env.Products      # Products
        self.F = env.F; self.Vehicles = env.Vehicles      # Fleet
        self.T = env.T
        self.Horizon = env.Horizon
    
        self.s_params = env.stochastic_parameters
        self.others = env.other_env_params

        self.sample_path_window_size = {}
        
        self.W_t = {t:{'q':{}, 'p': {}, 'd': {}, 'h': {}} for t in self.Horizon}

        if self.others['look_ahead']:
            self.S = env.S                                          # Number of sample paths
            self.Samples = env.Samples
            self.LA_horizon = env.LA_horizon                        # Look-ahead time window's size (includes current period)
            self.sample_paths = {t:{'q':{}, 'p': {}, 'd': {}, 'h': {}} for t in self.Horizon}
        
        ### historical log parameters ###
        if self.others['historical']:  
            self.hist_window = env.hist_window      # historical window 
            self.TW = env.TW
            self.historical = env.historical
            self.historical_data = {t:{'q':{}, 'p': {}, 'd': {}, 'h': {}} for t in self.Horizon}
        else:
            self.TW = self.Horizon


    def gen_ages(self):
        '''
        O_k: (dict) maximum days that k \in K can be held in inventory
        '''
        # Maximum days that product k can be held in inventory before rotting
        max_age = self.T
        self.O_k = {k:randint(1,max_age) for k in self.Products}

        return self.O_k 
    

    def gen_routing_costs(self):
        '''
        c: (dict) transportation cost between nodes i \in V and j \in V
        '''
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid), randint(0, size_grid)) for i in self.V}

        # Transportation cost between nodes i and j, estimated using euclidean distance
        c = {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in self.V for j in self.V if i!=j} 

        return c


    def gen_availabilities(self):
        '''
        M_kt: (dict) subset of suppliers that offer k \in K on t \in T
        K_it: (dict) subset of products offered by i \in M on t \in T
        '''
        self.M_kt = {}
        # In each time period, for each product
        for k in self.Products:
            for t in self.TW:
                # Random number of suppliers that offer k in t
                sup = randint(1, self.M+1)
                self.M_kt[k,t] = list(self.Suppliers)
                # Random suppliers are removed from subset, regarding {sup}
                for ss in range(self.M - sup):
                    a = int(randint(0, len(self.M_kt[k,t])))
                    del self.M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        self.K_it = {(i,t):[k for k in self.Products if i in self.M_kt[k,t]] for i in self.Suppliers for t in self.TW}

        return self.M_kt, self.K_it


    def gen_quantities(self, **kwargs):
        '''
        q_t: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        '''
        if kwargs['distribution'] == 'c_uniform':

            # Historic values
            if self.others['historical'] != False and ('q' in self.others['historical'] or '*' in self.others['historical']):
                self.historical_data[0]['q'] = {(i,k):[round(uniform(kwargs['min'], kwargs['max']),2) if i in self.M_kt[k,t] else 0 for t in self.historical] for i in self.Suppliers for k in self.Products}

            sample_path_window_size = copy(self.LA_horizon)
            for t in self.Horizon:   
                values_day_0 = {(i,k): round(uniform(kwargs['min'], kwargs['max']),2) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products}

                if t + self.LA_horizon > self.T:
                    sample_path_window_size = self.T - t
                self.sample_path_window_size[t] = sample_path_window_size

                # Generating sample-paths
                for day in range(sample_path_window_size):
                    for sample in self.Samples:
                        if day == 0 and (self.s_params == False or ('q' not in self.s_params and '*' not in self.s_params)):
                            self.sample_paths[t]['q'][day,sample] = values_day_0
                        else:
                            self.sample_paths[t]['q'][day,sample] = {(i,k): self.sim([self.historical_data[t]['q'][i,k][obs] for obs in range(len(self.historical_data[t]['q'][i,k])) if self.historical_data[t]['q'][i,k][obs] > 0]) if i in self.M_kt[k,t+day] else 0 for i in self.Suppliers for k in self.Products}
                
                # Generating random variable realization
                if self.s_params != False and ('q' in self.s_params or '*' in self.s_params):
                    self.W_t[t]['q'] = {(i,k): round(uniform(kwargs['min'], kwargs['max']),2) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products}
                else:
                    self.W_t[t]['q'] = values_day_0
                
                # Updating historical values
                if t < self.T - 1:
                    for i in self.Suppliers:
                        for k in self.Products:
                            self.historical_data[t+1]['q'][i,k] = self.historical_data[t]['q'][i,k] + [self.W_t[t]['q'][i,k]]
    
        
    def gen_demand(self, **kwargs):
        '''
        d_t: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        '''
        if kwargs['distribution'] == 'log-normal':

            # Historic values
            if self.others['historical'] != False and ('d' in self.others['historical'] or '*' in self.others['historical']):
                self.historical_data[0]['d'] = {k:[round(lognormal(kwargs['mean'], kwargs['stdev']),2) for t in self.historical] for k in self.Products}

            sample_path_window_size = copy(self.LA_horizon)
            for t in self.Horizon:   
                values_day_0 = {k: round(lognormal(kwargs['mean'], kwargs['stdev']),2) for k in self.Products}

                if t + self.LA_horizon > self.T:
                    sample_path_window_size = self.T - t

                # Generating sample-paths
                for day in range(sample_path_window_size):
                    for sample in self.Samples:
                        if day == 0 and (self.s_params == False or ('d' not in self.s_params and '*' not in self.s_params)):
                            self.sample_paths[t]['d'][day,sample] = values_day_0
                        else:
                            self.sample_paths[t]['d'][day,sample] = {k: self.sim(self.historical_data[t]['d'][k]) for k in self.Products}
                        
                # Generating random variable realization
                if self.s_params != False and ('q' in self.s_params or '*' in self.s_params):
                    self.W_t[t]['d'] = {k: round(lognormal(kwargs['mean'], kwargs['stdev']),2) for k in self.Products}
                else:
                    self.W_t[t]['d'] = values_day_0
                
                # Updating historical values
                if t < self.T - 1:
                    for k in self.Products:
                        self.historical_data[t+1]['d'][k] = self.historical_data[t]['d'][k] + [self.W_t[t]['d'][k]]



    def gen_p_price(self, **kwargs):
        '''
        p_t: (dict) price of k \in K offered by supplier i \in M on t \in T
        '''
        self.p_t = {t:{} for t in self.Horizon}

        if kwargs['distribution'] == 'd_uniform':

            # Historic values
            if self.others['historical'] != False and ('p' in self.others['historical'] or '*' in self.others['historical']):
                self.historical_data[0]['p'] = {(i,k):[randint(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 1000 for t in self.historical] for i in self.Suppliers for k in self.Products}

            sample_path_window_size = copy(self.LA_horizon)
            for t in self.Horizon:
                '''
                SAMPLE PATH GENERATION FOR STOCHASTIC VERSION OF PURCHASING PRICES
                values_day_0 = {(i,k): uniform(kwargs['min'], kwargs['max']) if i in self.M_kt[k,0] else 0 for i in self.Suppliers for k in self.Products}

                if t + self.LA_horizon > self.T:
                    sample_path_window_size = self.T - t

                # Generating sample-paths
                for day in range(sample_path_window_size):
                    for sample in self.Samples:
                        if day == 0 and (self.s_params == False or ('p' not in self.s_params and '*' not in self.s_params)):
                            self.sample_paths[t]['p'][day,sample] = values_day_0
                        else:
                            self.sample_paths[t]['p'][day,sample] = {(i,k): self.sim(self.historical_data[t]['q'][i,k]) if i in self.M_kt[k,day] else 0 for i in self.Suppliers for k in self.Products}
                        
                # Generating random variable realization
                if self.s_params != False and ('p' in self.s_params or '*' in self.s_params):
                    self.W_t[t]['p'] = {(i,k): uniform(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products}
                else:
                    self.W_t[t]['p'] = values_day_0
                
                # Updating historical values
                if t < self.T - 1:
                    for i in self.Suppliers:
                        for k in self.Products:
                            self.historical_data[t+1]['p'][i,k] = self.historical_data[t]['p'][i,k] + [self.W_t[t]['p'][i,k]] 
                '''
                # Genrating realizations
                self.p_t[t] = {(i,k): randint(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products}   
                self.W_t[t]['p'] = self.p_t[t]

                # Updating historical values
                if t < self.T - 1:
                    for i in self.Suppliers:
                        for k in self.Products:
                            self.historical_data[t+1]['p'][i,k] = self.historical_data[t]['p'][i,k] + [self.W_t[t]['p'][i,k]]
            
            return self.p_t

    
    def gen_h_cost(self, **kwargs):
        '''
        h_t: (dict) holding cost of k \in K on t \in T
        '''
        self.h_t = {t:{} for t in self.Horizon}

        if kwargs['distribution'] == 'd_uniform':

            # Historic values
            if self.others['historical'] != False and ('h' in self.others['historical'] or '*' in self.others['historical']):
                self.historical_data[0]['h'] = {k:[randint(kwargs['min'], kwargs['max']) for t in self.historical] for k in self.Products}

            sample_path_window_size = copy(self.LA_horizon)
            for t in self.Horizon:   
                '''
                SAMPLE PATH GENERATION FOR STOCHASTIC VERSION OF HOLDING COSTS

                values_day_0 = {k:self.sim(self.historical_data[t]['h'][k]) for k in self.Products}

                if t + self.LA_horizon > self.T:
                    sample_path_window_size = self.T - t

                # Generating sample-paths
                for day in range(sample_path_window_size):
                    for sample in self.Samples:
                        if day == 0 and (self.s_params == False or ('h' not in self.s_params and '*' not in self.s_params)):
                            self.sample_paths[t]['h'][day,sample] = values_day_0
                        else:
                            self.sample_paths[t]['h'][day,sample] = {k:self.sim(self.historical_data[t]['h'][k]) for k in self.Products}
                
                # Generating random variable realization
                if self.s_params != False and ('h' in self.s_params or '*' in self.s_params):
                    self.W_t[t]['h'] = {k:randint(kwargs['min'], kwargs['max']) for k in self.Products}
                else:
                    self.W_t[t]['h'] = values_day_0
                
                # Updating historical values
                if t < self.T - 1:
                    for k in self.Products:
                        self.historical_data[t+1]['h'][k] = self.historical_data[t]['h'][k] + [self.W_t[t]['h'][k]]
                '''
                # Genrating realizations
                self.h_t[t] = {k:randint(kwargs['min'], kwargs['max']) for k in self.Products} 
                self.W_t[t]['h'] = self.h_t[t]
            
                # Updating historical values
                if t < self.T - 1:
                    for k in self.Products:
                        self.historical_data[t+1]['h'][k] = self.historical_data[t]['h'][k] + [self.W_t[t]['h'][k]]
            
            return self.h_t

    # Auxuliary sample value generator function
    def sim(self, hist):
        ''' 
        Sample value generator function.
        Returns a generated random number using acceptance-rejection method.
        Parameters:
        - hist: (list) historical dataset that is used as an empirical distribution for
                the random number generation
        '''
        Te = len(hist)
        sorted_data = sorted(hist)
        
        prob, value = [], []
        for t in range(Te):
            prob.append((t+1)/Te)
            value.append(sorted_data[t])
        
        # Generates uniform random value for acceptance-rejection testing
        U = random()
        # Tests if the uniform random falls under the empirical distribution
        test = [i>U for i in prob]  
        # Takes the first accepted value
        sample = value[test.index(True)]
        
        return sample





    

    