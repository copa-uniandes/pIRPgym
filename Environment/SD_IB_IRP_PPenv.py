
"""
@author: juanbeta

TODO
! stochastic parameters

WARNINGS:
! Q parameter: Not well defined
! Check HOLDING COST (TIMING)

"""

################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time; from termcolor import colored
from random import random, seed, randint, shuffle

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

### Gym & OR-Gym
import gym; #from gym import spaces
from or_gym import utils

### Instance generation
from InstanceGenerator import instance_generator

################################ Description ################################
'''
State (S_t): The state according to Powell (three components): 
    - Physical State (R_t):
        state:  Current available inventory (!*): (dict)  Inventory of product k \in K of age o \in O_k
                When backlogs are activated, will appear under age 'B'
    - Other deterministic info (Z_t):
        p: Prices: (dict) Price of product k \in K at supplier i \in M
        q: Available quantities: (dict) Available quantity of product k \in K at supplier i \in M
        h: Holding cost: (dict) Holding cost of product k \in K
        historical_data: (dict) historical log of information (optional)
    - Belief State (B_t):
        sample_paths: Simulated sample paths (optional)

Action (X_t): The action can be seen as a three level-decision. These are the three layers:
    1. Routes to visit the selected suppliers
    2. Quantities to purchase on each supplier
    3. Demand compliance plan, dispatch decision
    4. (Optional) Backlogs compliance
    
    Accordingly, the action will be a list composed as follows:
    X = [routes, purchase, demand_compliance, backorders]
        - routes: (list) list of lists, each with the nodes visited on the route (including departure and arriving to the depot)
        - purchase: (dict) Units to purchase of product k \in K at supplier i \in M
        - demand_compliance: (dict) Units of product k in K of age o \in O_k used to satisfy the demand 
        - backlogs_compliance: (dict) Units of product k in K of age o \in O_k used to satisfy the backlogs


Exogenous information (W): The stochastic factors considered on the environment:
    Demand (dict) (*): Key k 
    Prices (dict) (*): Keys (i,k)
    Available quantities (dict) (*): Keys (i,k)
    Holding cost (dict) (*): Key k

(!*) Available inventory at the decision time. Never age 0 inventory.       
(*) Varying the stochastic factors might be of interest. Therefore, the deterministic factors
    will be under Z_t and stochastic factors will be generated and presented in the W_t
'''

################################## Steroid IRP class ##################################

class steroid_IRP(gym.Env): 
    '''
    Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products environment
    
    INITIALIZATION
    Look-ahead approximation: Generation of sample paths (look_ahead = ['d']):
    1. List of parameters to be forecasted on the look-ahead approximation ['d', 'p', ...]
    2. List with '*' to generate foreecasts for all parameters
    3. False for no sample path generation
    Related parameters:
        - S: Number of sample paths
        - LA_horizon: Number of look-ahead periods
            
    historical data: Generation or usage of historical data (historical_data = ['d'])   
    Three historical data options:
    1.  ['d', 'p', ...]: List with the parameters the historical info will be generated for
    2.  ['*']: historical info generated for all parameters
    3.  False: No historical data will be stored or used
    Related parameter:
        - hist_window: Initial log size (time periods)
    
    backorders: Catch unsatisfied demand (backorders = 'backorders'):
    1. 'backorders': Demand may be not fully satisfied. Non-complied orders will be automatically fullfilled at an extra-cost
    2. 'backlogs': Demand may be not fully satisfied. Non-complied orders will be registered and kept track of on age 'B'
    3. False: All demand must be fullfilled
    Related parameter:
        - back_o_cost = 600
        - back_l_cost = 20 
    
    stochastic_parameters: Which of the parameters are not known when the action is performed
    1.  ['d', 'p', ...]: List with the parameters that are stochastic
    2.  ['*']: All parameters are stochastic (h,p,d,q)
    3.  False: All parameters are deterministic
    
    PARAMETERS
    look_ahead = ['*']: Generate sample paths for look-ahead approximation
    historical_data = ['*']: Use of historicalal data
    backorders = False: Backorders
    rd_seed = 0: Seed for random number generation
    **kwargs: 
        M = 10: Number of suppliers
        K = 10: Number of Products
        F = 2:  Number of vehicles on the fleet
        T = 6:  Number of decision periods
        
        wh_cap = 1e9: Warehouse capacity
        penalization_cost: Penalization costs for RL
        
        S = 4:  Number of sample paths 
        LA_horizon = 5: Number of look-ahead periods
        lambda1 = 0.5: Controls demand, assures feasibility
        
    Two main functions:
    -   reset(return_state = False)
    -   step(action)
    '''
    
    # Initialization method
    def __init__(self, look_ahead = ['*'], historical_data = ['*'], backorders = False,
                 stochastic_parameters = False, **kwargs):
        
        ### Main parameters ###
        self.M = 10                                     # Suppliers
        self.K = 10                                     # Products
        self.F = 4                                      # Fleet
        self.T = 7                                  
        
        ### Other parameters ### 
        self.wh_cap = 1e9                               # Warehouse capacity

        self.Q = 1000 # TODO !!!!!!!!
        self.stochastic_parameters = stochastic_parameters
        
        ### Look-ahead parameters ###
        if look_ahead:    
            self.S = 4              # Number of sample paths
            self.LA_horizon = 5     # Look-ahead time window's size (includes current period)
        
        ### historical log parameters ###
        if historical_data:        
            self.hist_window = 40       # historical window

        ### Backorders parameters ###
        if backorders == 'backorders':
            self.back_o_cost = 600
        elif backorders == 'backlogs':
            self.back_l_cost = 500

        ### Custom configurations ###
        utils.assign_env_config(self, kwargs)
        self.gen_sets()

        ### State space ###
        # Physical state
        self.state = {}     # Inventory
        
        ### Extra information ###
        self.other_env_params = {'look_ahead':look_ahead, 'historical': historical_data, 'backorders': backorders}


    # Auxiliary method: Generate iterables of sets
    def gen_sets(self):
    
        self.Suppliers = range(1,self.M + 1);  self.V = range(self.M + 1)
        self.Products = range(self.K)
        self.Vehicles = range(self.F)
        self.Horizon = range(self.T)

        if self.other_env_params['look_ahead']:
            self.Samples = range(self.S)

        if self.other_env_params['historical']:
            self.TW = range(-self.hist_window, self.T)
            self.historical = range(-self.hist_window, 0)
        else:
            self.TW = self.Horizon


    # Reseting the environment
    def reset(self, return_state = False, rd_seed = 0, **kwargs):
        '''
        Reseting the environment. Genrate or upload the instance.
        PARAMETER:
        return_state: Indicates whether the state is returned
         
        '''   
        self.t = 0
        
        # Generate parameters with the instance generator
        generator = instance_generator(self, self.rd_seed)

        self.O_k = generator.gen_ages()
        self.c = generator.gen_routing_costs()

        self.h_t = generator.gen_h_costs()

        self.M_kt, self.K_it = generator.gen_availabilities()
        self.q_t = generator.gen_quantities()
        self.p_t = generator.gen_p_prices()
        self.d_t = generator.gen_demand()

        ## State ##
        self.state = {(k,o):0   for k in self.Products for o in self.Ages[k]}
        if self.others['backorders'] == 'backlogs':
            for k in self.Products:
                self.state[k,'B'] = 0  
        
        self.h = {k: self.h_t[k,self.t] for k in self.Products}
        self.q = {(i,k): self.q_t[i,k,self.t] for i in self.Suppliers for k in self.Products}
        self.p = {(i,k): self.p_t[i,k,self.t] for i in self.Suppliers for k in self.Products}
        self.d = {k: self.d_t[k,self.t] for k in self.Products}







            # General parameter
            self.gen_instance_data()

            # Look-ahead, sample paths
            if self.others['look_ahead']:
                self.sample_path_window_size = copy(self.LA_horizon)
                self.gen_sample_paths()                        


        
    
    # Step 
    def step(self, action, validate_action = False, warnings = True):
        if validate_action:
            self.action_validity(action)

        # Exogenous information realization 
        W = self.gen_exog_info_W()

        if self.stochastic_parameters != False:
            self.q = W['q'];    self.p = W['p'];     self.d = W['d'];   self.h = W['h']
            real_action = self.get_real_actions(action)
        else:
            real_action = action

        # Inventory dynamics
        s_tprime, reward, back_orders = self.transition_function(real_action, W, warnings)

        # Reward
        transport_cost, purchase_cost, holding_cost, backorders_cost = self.compute_costs(real_action, s_tprime)
        reward += transport_cost + purchase_cost + holding_cost + backorders_cost

        # Time step update and termination check
        self.t += 1
        done = self.check_termination(s_tprime)
        _ = {}

        # State update
        if not done:
            self.update_state(s_tprime)
    
            # EXTRA INFORMATION TO BE RETURNED
            _ = {'p': self.p, 'q': self.q, 'h': self.h, 'd': self.d, 'backorders': back_orders}
            if self.others['historical']:
                _['historical_info'] = self.historical_data
            if self.others['look_ahead']:
                _['sample_paths'] = self.sample_paths

            
        return self.state, reward, done, _
    
    
    def action_validity(self, action):
        routes, purchase, demand_compliance = action[:3]
        if self.others['backorders'] == 'backlogs':   back_o_compliance = action[3]
        valid = True
        error_msg = ''
        
        # Routing check
        assert not len(routes) > self.F, 'The number of routes exceedes the number of vehicles'

        for route in routes:
            assert not (route[0] != 0 or route[-1] != 0), \
                'Routes not valid, must start and end at the depot'

            route_capacity = sum(purchase[node,k] for k in self.Products for node in route[1:-2])
            assert not route_capacity > self.Q, \
                "Purchased items exceed vehicle's capacity"

            assert not len(set(route)) != len(route) - 1, \
                'Suppliers can only be visited once by a route'

            for i in range(len(route)):
                assert not route[i] not in self.V, \
                    'Route must be composed of existing suppliers' 
            
        # Purchase
        for i in self.Suppliers:
            for k in self.Products:
                assert not purchase[i,k] > self.q[i,k], \
                    f"Purchased quantities exceed suppliers' available quantities  ({i},{k})"
        
        # Demand_compliance
        for k in self.Products:
            assert not (self.others['backorders'] != 'backlogs' and demand_compliance[k,0] > sum(purchase[i,k] for i in self.Suppliers)), \
                f'Demand compliance with purchased items of product {k} exceed the purchase'

            assert not (self.others['backorders'] == 'backlogs' and demand_compliance[k,0] + back_o_compliance[k,0] > sum(purchase[i,k] for i in self.Suppliers)), \
                f'Demand/backlogs compliance with purchased items of product {k} exceed the purchase'

            assert not sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) > self.d[k], \
                f'Trying to comply a non-existing demand of product {k}' 
            
            for o in range(1, self.O_k[k] + 1):
                assert not (self.others['backorders'] != 'backlogs' and demand_compliance[k,o] > self.state[k,o]), \
                    f'Demand compliance with inventory items exceed the stored items  ({k},{o})' 
                
                assert not (self.others['backorders'] == 'backlogs' and demand_compliance[k,o] + back_o_compliance[k,o] > self.state[k,o]), \
                    f'Demand/Backlogs compliance with inventory items exceed the stored items ({k},{o})'

        # backlogs
        if self.others['backorders'] == 'backlogs':
            for k in self.Products:
                assert not sum(back_o_compliance[k,o] for o in range(self.O_k[k])) > self.state[k,'B'], \
                    f'Trying to comply a non-existing backlog of product {k}'
        
        elif self.others['backorders'] == False:
            for k in self.Products:
                assert not sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) < self.d[k], \
                    f'Demand of product {k} was not fulfilled'


    def get_real_actions(self, action):
        '''
        When some parameters are stochastic, the chosen action might not be feasible.
        Therefore, an aditional intra-step computation must be made and andjustments 
        on the action might be necessary

        '''
        purchase, demand_compliance = action[1:3]

        # The purchase exceeds the available quantities of the suppliers
        real_purchase = {(i,k): min(purchase[i,k], self.q[i,k]) for i in self.Suppliers for k in self.Products}

        real_demand_compliance = copy(demand_compliance)
        for k in self.suppliers:
            # The demand is lower than the demand compliance plan 
            if sum(real_demand_compliance[k,o] for o in range(self.O_k[k] + 1)) > self.d[k]:
                age = self.O_k[k]
                diff = sum(real_demand_compliance[k,o] for o in range(self.O_k[k] + 1)) - self.d[k]
                while diff > 0:
                    if real_demand_compliance[k,age] < diff:
                        diff -= real_demand_compliance[k,age]
                        real_demand_compliance[k,age] = 0
                        age -= 1
                    else:
                        diff = -5
                        real_demand_compliance[k,age] -= diff

            # The demand compliance of purchased items differs from the purchase 
            real_demand_compliance[k,0] = min(real_demand_compliance[k,0], sum(real_purchase[i,k] for i in self.Suppliers))

        real_action = [action[0], real_purchase, real_demand_compliance]

        return real_action


    # Compute costs of a given procurement plan for a given day
    def compute_costs(self, action, s_tprime):
        routes, purchase, demand_compliance = action[:3]
        if self.others['backorders'] == 'backlogs':   back_o_compliance = action[3]

        transport_cost = 0
        for route in routes:
            transport_cost += sum(self.c[route[i], route[i + 1]] for i in range(len(route) - 1))
        
        purchase_cost = sum(purchase[i,k] * self.p[i,k]   for i in self.Suppliers for k in self.Products)
        
        # TODO!!!!!
        holding_cost = sum(sum(s_tprime[k,o] for o in range(1, self.O_k[k] + 1)) * self.h[k] for k in self.Products)

        backorders_cost = 0
        if self.others['backorders'] == 'backorders':
            backorders = round(sum(max(self.d[k] - sum(demand_compliance[k,o] for o in range(self.O_k[k]+1)),0) for k in self.Products),1)
            print(f'backorders: {backorders}')
            backorders_cost = backorders * self.back_o_cost
        
        elif self.others['backorders'] == 'backlogs':
            backorders_cost = sum(s_tprime[k,'B'] for k in self.Products) * self.back_l_cost

        return transport_cost, purchase_cost, holding_cost, backorders_cost
            
    
    # Inventory dynamics of the environment
    def transition_function(self, real_action, W, warnings):
        purchase, demand_compliance = real_action[1:3]
        # backlogs
        if self.others['backorders'] == 'backlogs':
            back_o_compliance = real_action[3]
        inventory = deepcopy(self.state)
        reward  = 0
        back_orders = {}

        # Inventory update
        for k in self.Products:
            inventory[k,1] = round(sum(purchase[i,k] for i in self.Suppliers) - demand_compliance[k,0],1)

            max_age = self.O_k[k]
            if max_age > 1:
                for o in range(2, max_age + 1):
                        inventory[k,o] = round(self.state[k,o - 1] - demand_compliance[k,o - 1],1)
            
            if self.others['backorders'] == 'backorders' and sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) < W['d'][k]:
                back_orders[k] = W['d'][k] - sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1))

            if self.others['backorders'] == 'backlogs':
                new_backlogs = round(max(self.W['d'][k] - sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)),0),1)
                inventory[k,'B'] = round(self.state[k,'B'] + new_backlogs - sum(back_o_compliance[k,o] for o in range(self.O_k[k]+1)),1)

            # Factibility checks         
            if warnings:
                if self.state[k, max_age] - demand_compliance[k,max_age] > 0:
                    # reward += self.penalization_cost
                    print(colored(f'Warning! {self.state[k, max_age] - demand_compliance[k,max_age]} units of {k} were lost due to perishability','yellow'))
    

                if sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) < W['d'][k]:
                    print(colored(f'Warning! Demand of product {k} was not fulfilled', 'yellow'))

            # if sum(inventory[k,o] for k in self.Products for o in range(self.O_k[k] + 1)) > self.wh_cap:
            #     reward += self.penalization_cost
            #     print(f'Warning! Capacity of the whareouse exceeded')

        return inventory, reward, back_orders


    # Checking for episode's termination
    def check_termination(self, s_tprime):
        done = False

        # Time-step limit
        if self.hor_typ:
            done = self.t >= self.T
         
        # # Exceedes wharehouse capacitiy
        # if sum(s_tprime[k,o] for k in self.Products for o in range(1, self.O_k[k] + 1)) >= self.wh_cap:
        #     done = True

        return done

    def update_state(self, s_tprime):
        # Update historicalals
        for k in self.Products:
            for i in self.Suppliers:
                if 'p' in self.others['historical']  or '*' in self.others['historical']:
                    self.historical_data['p'][i,k].append(self.p[i,k])
                if 'q' in self.others['historical']  or '*' in self.others['historical']:
                    self.historical_data['q'][i,k].append(self.q[i,k])
            if 'h' in self.others['historical']  or '*' in self.others['historical']:
                self.historical_data['h'][k].append(self.h[k])
            if 'd' in self.others['historical']  or '*' in self.others['historical']:
                self.historical_data['d'][k].append(self.d[k])

        # Update state
        if self.hor_typ:
            self.p = {(i,k): self.p_t[i,k,self.t] for i in self.Suppliers for k in self.Products}
            self.q = {(i,k): self.q_t[i,k,self.t] for i in self.Suppliers for k in self.Products}
            self.h = {k: self.h_t[k,self.t] for k in self.Products}
            self.d = {k: self.d_t[k,self.t] for k in self.Products}
        else:
            self.gen_realization()

        self.state = s_tprime

        # Update sample-paths
        self.gen_sample_paths()
     
        
    # Generates exogenous information vector W (stochastic realizations for each random variable)
    def gen_exog_info_W(self):
        W = {}
        if self.stochastic_parameters != False and 'h' in self.stochastic_parameters:
            W['h'] = {k:randint(self.min_hprice, self.max_hprice) for k in self.Products}
        else:
            W['h'] = self.h
        
        M_k = {}
        for k in self.Products:
            sup = randint(1, self.M)
            M_k[k] = list(self.Suppliers)
            for ss in range(self.M - sup):
                a = int(randint(0, len(M_k[k])-1))
                del M_k[k][a]
        
        K_it = {i:[k for k in self.Products if i in M_k[k]] for i in self.Suppliers}
        
        if self.stochastic_parameters != False and 'q' in self.stochastic_parameters:
            W['q'] = {(i,k):randint(1,15) if i in self.M_k[k] else 0 for i in self.Suppliers for k in self.Products}
        else:
            W['q'] = self.q

        if self.stochastic_parameters != False and 'p' in self.stochastic_parameters:
            W['p'] = {(i,k):randint(1,500) if i in M_k[k] else 1000 for i in self.Suppliers for k in self.Products}
        else:
            W['p'] = self.p

        if self.stochastic_parameters != False and 'd' in self.stochastic_parameters:
            W['d'] = {k:round((self.lambda1 * max([W['q'][i,k] for i in self.Suppliers]) + (1-self.lambda1)*sum([W['q'][i,k] for i in self.Suppliers])),1) for k in self.Products} 
        else:
            W['d'] = self.d

        return W
    
    
    # Auxiliary function to manage historical and simulated data 
    def gen_instance_data(self):
        if type(self.others['historical']) == list: 
            self.gen_simulated_data()
   
        elif type(self.others['historical']) == str:  
            self.upload_historical_data()
        
        else:
            raise ValueError('historical information parameter value not valid')
                  
    
    # Generate historical and simulated stochastic parameters based on the requirement
    def gen_simulated_data(self):
        ''' 
        Simulated historicalal and sumulated data generator for quantities, prices and demand of products in each period.
        Generates:
            - h_t: (dict) holding cost of k \in K on t \in T
            - M_kt: (dict) subset of suppliers that offer k \in K on t \in T
            - K_it: (dict) subset of products offered by i \in M on t \in T
            - q_t: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            - p_t: (dict) price of k \in K offered by supplier i \in M on t \in T
            - d_t: (dict) demand of k \in K on t \in T
            - historical_data: (dict) with generated historical values
        '''
        self.historical_data = {}
        # Random holding cost of product k on t
        if 'h' in self.others['historical'] or  '*' in self.others['historical']:   
            self.historical_data['h'] = {k: [randint(self.min_hprice, self.max_hprice) for t in self.historical] for k in self.Products}
        self.h_t = {(k,t):randint(self.min_hprice, self.max_hprice) for k in self.Products for t in self.Horizon}
    
        self.M_kt = {}
        # In each time period, for each product
        for k in self.Products:
            for t in self.TW:
                # Random number of suppliers that offer k in t
                sup = randint(1, self.M)
                self.M_kt[k,t] = list(self.Suppliers)
                # Random suppliers are removed from subset, regarding {sup}
                for ss in range(self.M - sup):
                    a = int(randint(0, len(self.M_kt[k,t])-1))
                    del self.M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        self.K_it = {(i,t):[k for k in self.Products if i in self.M_kt[k,t]] for i in self.Suppliers for t in self.TW}
        
        # Random quantity of available product k, provided by supplier i on t
        if 'q' in self.others['historical'] or  '*' in self.others['historical']:
            self.historical_data['q']= {(i,k): [randint(1,15) if i in self.M_kt[k,t] else 0 for t in self.historical] for i in self.Suppliers for k in self.Products}
        self.q_t = {(i,k,t):randint(1,15) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products for t in self.Horizon}

        # Random price of available product k, provided by supplier i on t
        if 'p' in self.others['historical'] or  '*' in self.others['historical']:
            self.historical_data['p'] = {(i,k): [randint(1,500) if i in self.M_kt[k,t] else 1000 for t in self.historical] for i in self.Suppliers for k in self.Products for t in self.historical}
        self.p_t = {(i,k,t):randint(1,500) if i in self.M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products for t in self.Horizon}

        # Demand estimation based on quantities - ensuring feasibility, no backlogs
        if 'd' in self.others['historical'] or  '*' in self.others['historical']:
            self.historical_data['d'] = {(k):[(self.lambda1 * max([self.historical_data['q'][i,k][t] for i in self.Suppliers]) + (1-self.lambda1)*sum([self.historical_data['q'][i,k][t] for i in self.Suppliers])) for t in self.historical] for k in self.Products}
        self.d_t = {(k,t):round((self.lambda1 * max([self.q_t[i,k,t] for i in self.Suppliers]) + (1-self.lambda1)*sum([self.q_t[i,k,t] for i in self.Suppliers])),1) for k in self.Products for t in self.Horizon}
    
   
    # Auxuliary sample value generator function
    def sim(self, hist):
        ''' 
        Sample value generator function.
        Returns a generated random number using acceptance-rejection method.
        Parameters:
        - hist: (list) historicalal dataset that is used as an empirical distribution for
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
    
    
    # Sample paths generator function
    def gen_sample_paths(self):
        ''' 
        Sample paths generator function.
        Returns:
            - Q_s: (float) feasible vehicle capacity to use in rolling horizon model in sample path s \in Sam
            - M_kts: (dict) subset of suppliers that offer k \in K on t \in T in sample path s \in Sam
            - K_its: (dict) subset of products offered by i \in M on t \in T in sample path s \in Sam
            - q_s: (dict) quantity of k \in K offered by supplier i \in M on t \in T in sample path s \in Sam
            - p_s: (dict) price of k \in K offered by supplier i \in M on t \in T in sample path s \in Sam
            - dem_s: (dict) demand of k \in K on t \in T in sample path s \in Sam
            - 
            - F_s: (iter) set of vehicles in sample path s \in Sam
        Parameters:
            - hist_T: (int) number of periods that the historicalal datasets have information of
            - today: (int) current time period
        '''

        if self.hor_typ and self.t + self.LA_horizon > self.T:
                self.sample_path_window_size = self.T - self.t

        self.sample_paths = {}
        
        for s in self.Samples:
            # For each product, on each period chooses a random subset of suppliers that the product has had
            self.sample_paths[('M_k',s)] = {(k,t): [self.M_kt[k,tt] for tt in range(-self.hist_window + 1, self.t)][randint(-self.hist_window + 1, self.t - 1)] for k in self.Products for t in range(1, self.sample_path_window_size)}
            for k in self.Products:
                self.sample_paths[('M_k',s)][(k,0)] = self.M_kt[k, self.t]
            
            # Products offered by each supplier on each time period, based on M_kts
            self.sample_paths[('K_i',s)] = {(i,t): [k for k in self.Products if i in self.sample_paths[('M_k',s)][(k,t)]] \
                for i in self.Suppliers for t in range(1, self.sample_path_window_size)}
            for i in self.Suppliers:
               self.sample_paths[('K_i',s)][(k,0)] = self.K_it[i, self.t]
            

            # For each supplier and product, on each period chooses a quantity to offer using the sample value generator function
            #if 'q' in self.others['look_ahead']:
            self.sample_paths[('q',s)] = {(i,k,t): self.sim([self.historical_data['q'][i,k][tt] for tt in range(-self.hist_window + 1, self.t) if self.historical_data['q'][i,k][tt] > 0]) if i in self.sample_paths[('M_k',s)][(k,t)] else 0 \
                for i in self.Suppliers for k in self.Products for t in range(1, self.sample_path_window_size)}
            for i in self.Suppliers:
                for k in self.Products:
                    self.sample_paths[('q',s)][(i,k,0)] = self.q[i,k]
            
            # For each supplier and product, on each period chooses a price using the sample value generator function
            if 'p' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
                self.sample_paths[('p',s)] = {(i,k,t): self.sim([self.historical_data['p'][i,k][tt] for tt in range(-self.hist_window + 1, self.t) if self.historical_data['p'][i,k][tt] < 1000]) if i in self.sample_paths[('M_k',s)][(k,t)] else 1000 \
                    for i in self.Suppliers for k in self.Products for t in range(1, self.sample_path_window_size)}
                for i in self.Suppliers:
                    for k in self.Products:
                        self.sample_paths[('p',s)][i,k,0] = self.p[i,k]
            
            if 'h' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
                self.sample_paths[('h',s)] = {(k,t): self.sim(self.historical_data['h'][k]) for k in self.Products for t in range(1, self.sample_path_window_size)}
                for k in self.Products:
                    self.sample_paths[('h',s)][k,0] = self.h[k]
            
            # Estimates demand for each product, on each period, based on q_s
            if 'd' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
                self.sample_paths[('d',s)] = {(k,t): (self.lambda1 * max([self.sample_paths[('q',s)][(i,k,t)] for i in self.Suppliers]) + (1 - self.lambda1) * sum([self.sample_paths[('q',s)][(i,k,t)] \
                    for i in  self.Suppliers])) for k in self.Products for t in range(1, self.sample_path_window_size)}
                for k in self.Products:
                    self.sample_paths[('d',s)][k,0] = self.d[k]
            
            # Vehicle capacity estimation
            # if 'Q' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
            #     self.sample_paths[('Q',s)] = 1.2 * self.gen_Q()
            
            # Set of vehicles, based on estimated required vehicles
            # if 'F' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
            #     self.sample_paths[('F',s)] = int(sum(self.sample_paths[('d',s)].values())/self.sample_paths[('Q',s)]+1)


    # Simple function to visualize the inventory
    def print_inventory(self):
        max_O = max([self.O_k[k] for k in self.Products])
        listamax = [[self.state[k,o] for o in self.Ages[k]] for k in self.Products]
        df = pd.DataFrame(listamax, index=pd.Index([str(k) for k in self.Products], name='Products'),
        columns=pd.Index([str(o) for o in range(1, max_O + 1)], name='Ages'))

        return df


    # Printing a representation of the environment (repr(env))
    def __repr__(self):
        return f'Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance. V = {self.M}; K = {self.K}; F = {self.F}'


    ##################### EXTRA Non-funcitonal features #####################
    '''
    1. Uploading instance from .txt file
    
    def upload_instance(self, nombre, path = ''):
        
        #sys.path.insert(1, path)
        with open(nombre, "r") as f:
            
            linea1 = [x for x in next(f).split()];  linea1 = [x for x in next(f).split()] 
            Vertex = int(linea1[1])
            linea1 = [x for x in next(f).split()];  Products = int(linea1[1])
            linea1 = [x for x in next(f).split()];  Periods = int(linea1[1])
            linea1 = [x for x in next(f).split()];  linea1 = [x for x in next(f).split()] 
            Q = int(linea1[1])   
            linea1 = [x for x in next(f).split()]
            coor = {}
            for k in range(Vertex):
                linea1= [int(x) for x in next(f).split()];  coor[linea1[0]] = (linea1[1], linea1[2])   
            linea1 = [x for x in next(f).split()]  
            h = {}
            for k in range(Products):
                linea1= [int(x) for x in next(f).split()]
                for t in range(len(linea1)):  h[k,t] = linea1[t]    
            linea1 = [x for x in next(f).split()]
            d = {}
            for k in range(Products):
                linea1= [int(x) for x in next(f).split()]
                for t in range(len(linea1)):  d[k,t] = linea1[t]
            linea1 = [x for x in next(f).split()] 
            O_k = {}
            for k in range(Products):
                linea1= [int(x) for x in next(f).split()];  O_k[k] = linea1[1] 
            linea1 = [x for x in next(f).split()]
            Mk = {};  Km = {};  q = {};  p = {} 
            for t in range(Periods):
                for k in range(Products):
                    Mk[k,t] = []    
                linea1 = [x for x in next(f).split()] 
                for i in range(1, Vertex):
                    Km[i,t] = [];   linea = [int(x) for x in next(f).split()]  
                    KeyM = linea[0];   prod = linea[1];   con = 2 
                    while con < prod*3+2:
                        Mk[linea[con], t].append(KeyM);   p[(KeyM, linea[con],t)]=linea[con+1]
                        q[(KeyM, linea[con],t)]=linea[con+2];  Km[i,t].append(linea[con]);   con = con + 3
        
        self.M = Vertex;   self.Suppliers = range(1, self.M);   self.V = range(self.M)
        self.P = Products; self.Products = range(self.P)
        self.T = Periods;  self.Horizon = range(self.T)
 
        self.F, I_0, c  = self.extra_processing(coor)
        self.Vehicles = range(self.F)
        
        return O_k, c, Q, h, Mk, Km, q, p, d, I_0

    def extra_processing(self, coor):
        
        F = int(np.ceil(sum(self.d.values())/self.Q)); self.Vehicles = range(self.F)
        I_0 = {(k,o):0 for k in self.Products for o in range(1, self.O_k[k] + 1)} # Initial inventory level with an old > 1 
        
        # Travel cost between (i,j). It is the same for all time t
        c = {(i,j,t,v):round(np.sqrt(((coor[i][0]-coor[j][0])**2)+((coor[i][1]-coor[j][1])**2)),0) for v in self.Vehicles for t in self.Horizon for i in self.V for j in self.V if i!=j }
        
        return F, I_0, c
    

    2. Upload file with historical information 

    def upload_historical_data(self):  

        self.h_t =
        self.q_t =
        self.p_t =
        self.d_t =
    
    
    
    '''