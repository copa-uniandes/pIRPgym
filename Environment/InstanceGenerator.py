################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time
from random import random, seed, randint, shuffle, uniform

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

### Gym & OR-Gym
import gym; from gym import spaces
# TODO Check if import or_gym works
import utils


class instance_generator():

    def __init__(self, look_ahead = ['d'], stochastic_params = False, historical_data = ['*'],
                  backorders = 'backorders', stoch=True, **kwargs):
        
        ### Main parameters ###
        self.M = 10                                     # Suppliers
        self.K = 10                                     # Products
        self.F = 4                                      # Fleet
        self.T = 7        

        ### Look-ahead parameters ###
        if look_ahead:    
            self.S = 4              # Number of sample paths
            self.LA_horizon = 5     # Look-ahead time window's size (includes current period)
            self.s_path_window_sizes = {t:min(self.LA_horizon, self.T - t) for t in range(self.T)}
        
        ### historical log parameters ###
        if historical_data:        
            self.hist_window = 40       # historical window

        ### Backorders parameters ###
        if backorders == 'backorders':
            self.back_o_cost = 600
        elif backorders == 'backlogs':
            self.back_l_cost = 500

        ### Extra information ###
        self.other_params = {'look_ahead':look_ahead, 'historical': historical_data, 'backorders': backorders}
        self.s_params = stochastic_params

        ### Custom configurations ###
        utils.assign_env_config(self, kwargs)

        

    # Generates an instance with a given random seed
    def generate_instance(self, d_rd_seed, s_rd_seed, **kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        self.gen_sets()

        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}


        # Offer
        self.M_kt, self.K_it = offer.gen_availabilities(self)
        self.hist_q, self.W_q, self.s_paths_q = offer.gen_quantities(self, **kwargs['q_params'])
        if self.s_paths_q == None: del self.s_paths_q

        # Demand

        # Inventory

        # Routing




    # Auxiliary method: Generate iterables of sets
    def gen_sets(self):
        self.Suppliers = range(1,self.M + 1);  self.V = range(self.M + 1)
        self.Products = range(self.K)
        self.Vehicles = range(self.F)
        self.Horizon = range(self.T)

        if self.other_params['look_ahead']:
            self.Samples = range(self.S)

        if self.other_params['historical']:
            self.TW = range(-self.hist_window, self.T)
            self.historical = range(-self.hist_window, 0)
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
            
     
    def gen_stoch_historics(self, **kwargs):
        # Historic values
        if self.others['historical'] != False and ('q' in self.others['historical'] or '*' in self.others['historical']):
            self.historical_data[0]['q'] = {(i,k):[round(uniform(kwargs['min'], kwargs['max']),2) if i in self.M_kt[k,t] else 0 for t in self.historical] for i in self.Suppliers for k in self.Products}

        # Historic values
        if self.others['historical'] != False and ('d' in self.others['historical'] or '*' in self.others['historical']):
            self.historical_data[0]['d'] = {k:[round(lognormal(kwargs['mean'], kwargs['stdev']),2) for t in self.historical] for k in self.Products}


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
        seed(randint(0,1e9))
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






class costs():

    def __init__(self):
        pass


class demand():

    def __init__(self):
        pass

    def log_normal_demand(self, **kwargs):
        '''
        d_t: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        '''
        seed(self.stoch_rd_seed)
        if kwargs['distribution'] == 'log-normal':

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
                
                #seed(self.stoch_rd_seed + self.M * self.K + t)
                # Generating random variable realization
                if self.s_params != False and ('d' in self.s_params or '*' in self.s_params):
                    self.W_t[t]['d'] = {k: round(lognormal(kwargs['mean'], kwargs['stdev']),2) for k in self.Products}
                else:
                    self.W_t[t]['d'] = values_day_0
                
                # Updating historical values
                if t < self.T - 1:
                    for k in self.Products:
                        self.historical_data[t+1]['d'][k] = self.historical_data[t]['d'][k] + [self.W_t[t]['d'][k]] 


class offer():

    def __init__(self):
        pass

    def gen_availabilities(inst_gen: instance_generator):
        '''
        M_kt: (dict) subset of suppliers that offer k \in K on t \in T
        K_it: (dict) subset of products offered by i \in M on t \in T
        '''
        M_kt = {}
        # In each time period, for each product
        for k in inst_gen.Products:
            for t in inst_gen.TW:
                # Random number of suppliers that offer k in t
                sup = randint(1, inst_gen.M+1)
                M_kt[k,t] = list(inst_gen.Suppliers)
                # Random suppliers are removed from subset, regarding {sup}
                for ss in range(inst_gen.M - sup):
                    a = int(randint(0, len(M_kt[k,t])-1))
                    del M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        K_it = {(i,t):[k for k in inst_gen.Products if i in M_kt[k,t]] for i in inst_gen.Suppliers for t in inst_gen.TW}

        return M_kt, K_it
    

    #TODO rd_function parameter and the kwargs
    def gen_quantities(inst_gen: instance_generator, **kwargs):
        if kwargs['distribution'] == 'c_uniform':   rd_function = randint
        hist_q = offer.gen_hist_q(inst_gen, rd_function, kwargs)
        W_q, hist_q = offer.gen_W_q(inst_gen, hist_q, kwargs)
        s_paths_q = offer.gen_sp_q(inst_gen, hist_q, W_q, kwargs)

        return hist_q, W_q, s_paths_q 
               
    
    def gen_hist_q(inst_gen, rd_function, **kwargs):
        hist_q = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_q[0] = {(i,k):[round(rd_function(kwargs['r_f_params']),2) if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
        else:
            hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

        return hist_q


    def gen_W_q(inst_gen: instance_generator, hist_q, **kwargs):
        '''
        W_q: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        '''
        W_q = {}
        for t in inst_gen.Horizon:
            W_q[t] = {}   
            for i in inst_gen.Suppliers:
                for k in inst_gen.Products:
                    if i in inst_gen.M_kt[k,t]:
                        W_q[t][(i,k)] = round(kwargs['random_function'](kwargs['r_f_params']),2)
                    else:   W_q[t][(i,k)] = 0

                    if t < inst_gen.T - 1:
                        hist_q[t+1][i,k] = hist_q[t][i,k] + [W_q[t][i,k]]

        return W_q, hist_q

    
    def gen_sp_q(inst_gen, hist_q, W_q, **kwargs):
        if inst_gen.other_params['look_ahead'] != False and ('q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']): 
            s_paths_q = {}
            for t in inst_gen.Horizon: 
                s_paths_q[t] = {}
                for sample in inst_gen.Samples:
                    if inst_gen.s_params == False or ('q' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                        s_paths_q[t][0,sample] = W_q[t]
                    else:
                        s_paths_q[t][0,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t]['q'][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}
                    for day in range(1,inst_gen.sample_path_window_size[t]):
                        s_paths_q[t][day,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t]['q'][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

            return s_paths_q

        else:
            return None

    

class routes():

    def __init__(self):
        pass 


    def generate_grid(V): 
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid), randint(0, size_grid)) for i in V}
        return coor, V
    

    def euclidean_distance(coor, V):
        # Transportation cost between nodes i and j, estimated using euclidean distance
        return {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in V for j in V if i!=j}
    

    def euclidean_d_costs(self, V):
        return self.euclidean_distance(self.generate_grid(V))
