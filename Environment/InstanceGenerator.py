################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time; from termcolor import colored
from numpy.random import seed, randint, normal, lognormal

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

### Gym & OR-Gym
import gym; #from gym import spaces
from or_gym import utils


class instance_generator():

    def __init__(self, r_seed = False, **kwargs):
        
        if r_seed:
            seed(r_seed)
        
        ### Main parameters ###
        self.M = 10                                    # Suppliers
        self.K = 10                                    # Products
        self.F = 4                                     # Fleet
        self.T = 7
        
        ### historical log parameters ###
        self.hist_window = 40       # historical window

        utils.assign_env_config(self, kwargs)
        self.gen_iterables()

    def gen_iterables(self):

        self.Suppliers = range(1,self.M);  self.V = range(self.M)
        self.Products = range(self.K)
        self.Vehicles = range(self.F)
        self.Horizon = range(self.T)
        self.TW = range(-self.hist_window, self.T)
        self.historical = range(-self.hist_window, 0)
        self.historical_data = {}


    def gen_availabilities(self):

        self.M_kt = {}
        # In each time period, for each product
        for k in self.Products:
            for t in self.TW:
                # Random number of suppliers that offer k in t
                sup = randint(1, self.M)
                self.M_kt[k,t] = list(self.Suppliers)
                # Random suppliers are removed from subset, regarding {sup}
                for ss in range(self.M - sup):
                    print(len(self.M_kt[k,t])-1)
                    a = int(randint(0, len(self.M_kt[k,t])-1))
                    del self.M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        self.K_it = {(i,t):[k for k in self.Products if i in self.M_kt[k,t]] for i in self.Suppliers for t in self.TW}

    def gen_quantities(self, distribution = 'normal', **kwargs):
        self.q_t = {}

        if distribution == 'normal':
            self.historical_data['q'] = {(i,k): [normal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 0 for t in self.historical] for i in self.Suppliers for k in self.Products}
            self.q_t = {(i,k,t):normal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products for t in self.Horizon}
        elif distribution == 'log-normal':
            self.historical_data['q'] = {(i,k): [lognormal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 0 for t in self.historical] for i in self.Suppliers for k in self.Products}
            self.q_t = {(i,k,t): lognormal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products for t in self.Horizon}
        elif distribution == 'uniform':
            self.historical_data['q'] = {(i,k): [randint(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 0 for t in self.historical] for i in self.Suppliers for k in self.Products}
            self.q_t = {(i,k,t): randint(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products for t in self.Horizon}
    

    def gen_demand(self, distribution = 'normal', **kwargs):
        self.d_t = {}

        if distribution == 'normal':
            self.historical_data['d'] = {(k):[normal(kwargs['mean'], kwargs['stdev']) for t in self.historical] for k in self.Products}
            self.d_t = {(k,t): normal(kwargs['mean'], kwargs['stdev']) for k in self.Products for t in self.Horizon}
        elif distribution == 'log-normal':
            self.historical_data['d'] = {(k):[lognormal(kwargs['mean'], kwargs['stdev']) for t in self.historical] for k in self.Products}
            self.d_t = {(k,t): lognormal(kwargs['mean'], kwargs['stdev']) for k in self.Products for t in self.Horizon}
        elif distribution == 'uniform':
            self.historical_data['d'] = {(k):[randint(kwargs['min'], kwargs['max']) for t in self.historical] for k in self.Products}
            self.d_t = {(k,t): randint(kwargs['min'], kwargs['max']) for k in self.Products for t in self.Horizon}

    def gen_p_price(self, distribution = 'normal', **kwargs):
        self.p_t = {}
        
        if distribution == 'normal':
            self.historical_data['p'] = {(i,k): [normal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 1000 for t in self.historical] for i in self.Suppliers for k in self.Products for t in self.historical}
            self.p_t = {(i,k,t): normal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products for t in self.Horizon}
        elif distribution == 'log-normal':
            self.historical_data['p'] = {(i,k): [lognormal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 1000 for t in self.historical] for i in self.Suppliers for k in self.Products for t in self.historical}
            self.p_t = {(i,k,t):lognormal(kwargs['mean'], kwargs['stdev']) if i in self.M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products for t in self.Horizon}
        elif distribution == 'uniform':
            self.historical_data['p'] = {(i,k): [randint(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 1000 for t in self.historical] for i in self.Suppliers for k in self.Products for t in self.historical}
            self.p_t = {(i,k,t): randint(kwargs['min'], kwargs['max']) if i in self.M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products for t in self.Horizon}
    

    def gen_h_price(self, distribution = 'normal', **kwargs):
        self.h_t = {}

        if distribution == 'normal':
            self.historical_data['h'] = {k: [normal(kwargs['mean'], kwargs['stdev']) for t in self.historical] for k in self.Products}
            self.h_t = {(k,t): normal(kwargs['mean'], kwargs['stdev']) for k in self.Products for t in self.Horizon}
        elif distribution == 'log-normal':
            self.historical_data['h'] = {k: [lognormal(kwargs['mean'], kwargs['stdev']) for t in self.historical] for k in self.Products}
            self.h_t = {(k,t): lognormal(kwargs['mean'], kwargs['stdev']) for k in self.Products for t in self.Horizon}
        elif distribution == 'uniform':
            self.historical_data['h'] = {k: [randint(kwargs['min'], kwargs['max']) for t in self.historical] for k in self.Products}
            self.h_t = {(k,t): randint(kwargs['min'], kwargs['max']) for k in self.Products for t in self.Horizon}
    
    def conf_int_gen(self, **kwargs):

        self.gen_availabilities()
        self.gen_quantities(**kwargs['q'])
        self.gen_demand(**kwargs['d'])
        self.gen_p_price(**kwargs['p'])
        self.gen_h_price(**kwargs['h'])


    

    