################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import sys; import pandas as pd; import math; import numpy as np
import time
#from random import random, seed, randint, shuffle, uniform
from numpy.random import seed, random, randint, lognormal


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
            self.sp_window_sizes = {t:min(self.LA_horizon, self.T - t) for t in range(self.T)}
        
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
        # TODO Implement appropriate seed setting 
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

        self.hist_p, self.W_p, self.s_paths_p = offer.gen_prices(self, **kwargs['p_params'])
        if self.s_paths_p == None: del self.s_paths_p

        # Demand
        self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand(self, **kwargs['d_params'])
        if self.s_paths_d == None: del self.s_paths_d

        # Inventory
        self.hist_h, self.W_h = costs.gen_h_cost(self, **kwargs['h_params'])

        # Routing
        self.c = locations.euclidean_dist_costs(self.V)


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
        self.O_k = {k:randint(1,max_age+1) for k in self.Products}

        return self.O_k 
        

    # Auxuliary sample value generator function
    def sim(self, hist):
        ''' 
        Sample value generator function.
        Returns a generated random number using acceptance-rejection method.
        Parameters:
        - hist: (list) historical dataset that is used as an empirical distribution for
                the random number generation
        '''
        seed(randint(0,int(1e6)))
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

    ### Holding cost
    def gen_h_cost(inst_gen: instance_generator, **kwargs) -> tuple:
        if kwargs['distribution'] == 'd_uniform':   rd_function = randint
        hist_h = costs.gen_hist_h(inst_gen, rd_function, **kwargs)
        W_h, hist_h = costs.gen_W_h(inst_gen, rd_function, hist_h, **kwargs)

        return hist_h, W_h
    

    # Historic holding cost
    def gen_hist_h(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]: 
        hist_h = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('h' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_h[0] = {k:[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products}
        else:
            hist_h[0] = {k:[] for k in inst_gen.Products}

        return hist_h


    # Realized (real) holding cost
    def gen_W_h(inst_gen: instance_generator, rd_function, hist_h, **kwargs) -> tuple:
        '''
        W_h: (dict) holding cost of k \in K  on t \in T
        '''
        W_h = {}
        for t in inst_gen.Horizon:
            W_h[t] = {}   
            for k in inst_gen.Products:
                W_h[t][k] = round(rd_function(*kwargs['r_f_params']),2)

                if t < inst_gen.T - 1:
                    hist_h[t+1][k] = hist_h[t][k] + [W_h[t][k]]

        return W_h, hist_h
    

class demand():

    def __init__(self):
        pass

    ### Demand of products
    def gen_demand(inst_gen: instance_generator, **kwargs) -> tuple:
        if kwargs['distribution'] == 'log-normal':   rd_function = lognormal
        hist_d = demand.gen_hist_d(inst_gen, rd_function, **kwargs)
        W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)

        if 'd' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            s_paths_d = demand.gen_empiric_d_sp(inst_gen, hist_d, W_d)
            return hist_d, W_d, s_paths_d

        else:
            return hist_d, W_d, None
    
    # Historic demand
    def gen_hist_d(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]: 
        hist_d = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('d' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_d[0] = {k:[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products}
        else:
            hist_d[0] = {k:[] for k in inst_gen.Products}

        return hist_d


    # Realized (real) availabilities
    def gen_W_d(inst_gen: instance_generator, rd_function, hist_d, **kwargs) -> tuple:
        '''
        W_d: (dict) demand of k \in K  on t \in T
        '''
        W_d = {}
        for t in inst_gen.Horizon:
            W_d[t] = {}   
            for k in inst_gen.Products:
                W_d[t][k] = round(rd_function(*kwargs['r_f_params']),2)

                if t < inst_gen.T - 1:
                    hist_d[t+1][k] = hist_d[t][k] + [W_d[t][k]]

        return W_d, hist_d
    

    # Demand's sample paths
    def gen_empiric_d_sp(inst_gen: instance_generator, hist_d, W_d) -> dict[dict]:
        s_paths_d = {}
        for t in inst_gen.Horizon: 
            s_paths_d[t] = {}
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('d' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_d[t][0,sample] = W_d[t]
                else:
                    s_paths_d[t][0,sample] = {k: inst_gen.sim([hist_d[t][k][obs] for obs in range(len(hist_d[t][k])) if hist_d[t][k][obs] > 0]) for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_d[t][day,sample] = {k: inst_gen.sim([hist_d[t][k][obs] for obs in range(len(hist_d[t][k])) if hist_d[t][k][obs] > 0]) for k in inst_gen.Products}

        return s_paths_d
    

class offer():

    def __init__(self):
        pass
    
    ### Availabilty of products on suppliers
    def gen_availabilities(inst_gen: instance_generator) -> tuple:
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
                    a = int(randint(0, len(M_kt[k,t])))
                    del M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        K_it = {(i,t):[k for k in inst_gen.Products if i in M_kt[k,t]] for i in inst_gen.Suppliers for t in inst_gen.TW}

        return M_kt, K_it
    

    ### Available quantities of products on suppliers
    def gen_quantities(inst_gen: instance_generator, **kwargs) -> tuple:
        if kwargs['distribution'] == 'c_uniform':   rd_function = randint
        hist_q = offer.gen_hist_q(inst_gen, rd_function, **kwargs)
        W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)

        if 'q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            s_paths_q = offer.gen_empiric_q_sp(inst_gen, hist_q, W_q)
            return hist_q, W_q, s_paths_q 

        else:
            return hist_q, W_q, None


    # Historic availabilities
    def gen_hist_q(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]:
        hist_q = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_q[0] = {(i,k):[round(rd_function(*kwargs['r_f_params']),2) if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
        else:
            hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

        return hist_q

    
    # Realized (real) availabilities
    def gen_W_q(inst_gen: instance_generator, rd_function, hist_q, **kwargs) -> tuple:
        '''
        W_q: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        '''
        W_q = {}
        for t in inst_gen.Horizon:
            W_q[t] = {}   
            for i in inst_gen.Suppliers:
                for k in inst_gen.Products:
                    if i in inst_gen.M_kt[k,t]:
                        W_q[t][(i,k)] = round(rd_function(*kwargs['r_f_params']),2)
                    else:   W_q[t][(i,k)] = 0

                    if t < inst_gen.T - 1:
                        hist_q[t+1][i,k] = hist_q[t][i,k] + [W_q[t][i,k]]

        return W_q, hist_q


    # Availabilitie's sample paths
    def gen_empiric_q_sp(inst_gen: instance_generator, hist_q, W_q) -> dict[dict]:
        s_paths_q = {}
        for t in inst_gen.Horizon: 
            s_paths_q[t] = {}
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('q' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_q[t][0,sample] = W_q[t]
                else:
                    s_paths_q[t][0,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,0] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_q[t][day,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

        return s_paths_q


    ### Prices of products on suppliers
    def gen_prices(inst_gen: instance_generator, **kwargs) -> tuple:
        if kwargs['distribution'] == 'd_uniform':   rd_function = randint
        hist_p = offer.gen_hist_p(inst_gen, rd_function, **kwargs)
        W_p, hist_p = offer.gen_W_p(inst_gen, rd_function, hist_p, **kwargs)

        if 'p' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            s_paths_p = offer.gen_empiric_p_sp(inst_gen, hist_p, W_p)
            return hist_p, W_p, s_paths_p 

        else:
            return hist_p, W_p, None
    
    
    # Historic prices
    def gen_hist_p(inst_gen, rd_function, **kwargs) -> dict[dict]:
        hist_p = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('p' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_p[0] = {(i,k):[round(rd_function(*kwargs['r_f_params']),2) if i in inst_gen.M_kt[k,t] else 1000 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
        else:
            hist_p[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

        return hist_p


    # Realized (real) prices
    def gen_W_p(inst_gen: instance_generator, rd_function, hist_p, **kwargs) -> tuple:
        '''
        W_p: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        '''
        W_p = {}
        for t in inst_gen.Horizon:
            W_p[t] = {}   
            for i in inst_gen.Suppliers:
                for k in inst_gen.Products:
                    if i in inst_gen.M_kt[k,t]:
                        W_p[t][(i,k)] = round(rd_function(*kwargs['r_f_params']),2)
                    else:   W_p[t][(i,k)] = 1000

                    if t < inst_gen.T - 1:
                        hist_p[t+1][i,k] = hist_p[t][i,k] + [W_p[t][i,k]]

        return W_p, hist_p
    

    # Prices's sample paths
    def gen_empiric_p_sp(inst_gen: instance_generator, hist_p, W_p) -> dict[dict]:
        s_paths_p = {}
        for t in inst_gen.Horizon: 
            s_paths_p[t] = {}
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('p' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_p[t][0,sample] = W_p[t]
                else:
                    s_paths_p[t][0,sample] = {(i,k): inst_gen.sim([hist_p[t][i,k][obs] for obs in range(len(hist_p[t][i,k])) if hist_p[t][i,k][obs] < 1000]) if i in inst_gen.M_kt[k,0] else 1000 for i in inst_gen.Suppliers for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_p[t][day,sample] = {(i,k): inst_gen.sim([hist_p[t][i,k][obs] for obs in range(len(hist_p[t][i,k])) if hist_p[t][i,k][obs] < 1000]) if i in inst_gen.M_kt[k,t+day] else 1000 for i in inst_gen.Suppliers for k in inst_gen.Products}

        return s_paths_p
    

class locations():

    def __init__(self):
        pass 


    def generate_grid(V: range): 
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid+1), randint(0, size_grid+1)) for i in V}
        return coor, V
    

    def euclidean_distance(coor: dict, V: range):
        # Transportation cost between nodes i and j, estimated using euclidean distance
        return {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in V for j in V if i!=j}
    

    def euclidean_dist_costs(V: range):
        return locations.euclidean_distance(*locations.generate_grid(V))
