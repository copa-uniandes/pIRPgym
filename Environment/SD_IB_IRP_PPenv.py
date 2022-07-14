
"""
@author: juanbeta

TODO:
! FIX SAMPLE PATHS
! Check documentation

FUTURE WORK - Not completely developed:
- Instance_file uploading
- Continuous time horizon
- Historic_file uploading
- Instance_file exporting

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

################################ Description ################################
'''
State (S): The state according to Powell (three components): 
    - Physical State (R_t):
        state: Current available inventory (!*): (dict)  Inventory of k \in K of age o \in O_k
                When back-logs are activated, will appear under age 'B'
    - Other deterministic info (Z_t):
        p: Prices: (dict) Price of k \in K at i \in M
        q: Available quantities: (dict) Available quantities of k \in K at i \in M
        h: Holding cost: (dict) Holding cost of k \in K
        historic_data: (dict) Historic log of information (optional)
    - Belief State (B_t):
        sample_paths: Sample paths (optional)

Action (X): The action can be seen as a three level-decision. These are the three layers:
    1. Routes to visit selected suppliers
    2. Quantity to purchase on each supplier
    3. Demand complience plan, dispatch decision
    
    Accordingly, the action will be a list composed as follows:
    X = [routes, purchase, demand_complience]
        - routes (list): list of list with the nodes visited on a route (including departure and arriving to the depot)
        - purchase (dict): Units to purchase of k \in K at i \in M
        - demand_complience (dict): Units to use of k in K of age o \in O_k


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
    Time Horizon: Two time horizon types (horizon_type = 'episodic')
    1. 'episodic': Every episode (simulation) has a finite number of steps
        Related parameters:
            - T: Decision periods (time-steps)
    2. 'continuous': Neverending episodes
        Related parameters: 
            - gamma: Discount factor
    (For internal environment's processes: 1 for episodic, 0 for continouos)
    
    Look-ahead approximation: Generation of sample paths (look_ahead = ['d']):
    1. List of parameters to be forcasted on the look-ahead approximation ['d', 'p', ...]
    2. List with '*' to generate forecasts for all parameters
    3. False for no sample path generation
    Related parameters:
        - S: Number of sample paths
        - LA_horizon: Number of look-ahead periods
            
    Historic data: Generation or usage of historic data (historic_data = ['d'])   
    Three historic data options:
    1.  ['d', 'p', ...]: List with the parameters the historic info will be generated for
    2.  ['*']: Historic info generated for all parameters
    3.  !!! NOT DEVELOPED path: File path to be processed by upload_historic_data() 
    4.  False: No historic data will be used
    Related parameter:
        - hist_window: Initial log size (time periods)
    
    Back-orders: Catch unsatisfied demand (back_orders = False):
    1. 'back-orders': Demand can be not fully satisfied. Non-complied orders will be automatically fullfilled at an extra-cost
    2. 'back-logs': Demand can be not fully satisfied. Non-complied orders will be registered and kept track of on age 'B'
    3. False: All demand must be fullfilled
    Related parameter:
        - back_o_cost = 20 
    
    PARAMETERS
    env_init = 'episodic': Time horizon type {episodic, continouos} 
    look_ahead = ['d']: Generate sample paths for look-ahead approximation
    historic_data = ['d']: Use of historical data
    back_orders = False: Back orders
    rd_seed = 0: Seed for random number generation
    wd = True: Working directory path
    file_name = True: File name when uploading instances from .txt
    **kwargs: 
        M = 10: Number of suppliers
        K = 10: Number of Products
        F = 2:  Number of vehicles on the fleete
        T = 6:  Number of decision periods
        
        wh_cap = 1e9: Warehouse capacity
        min/max_sprice: Max and min selling prices (per m and k)
        min/max_hprice: Max and min holding cost (per k)
        penalization_cost: Penalization costs for RL
        
        S = 4:  Number of sample paths 
        LA_horizon = 5: Number of look-ahead periods
        lambda1 = 0.5: Controls demand, assures feasibility
        
    Two main functions:
    -   reset(return_state = False)
    -   step(action)
    '''
    
    # Initialization method
    def __init__(self, horizon_type = 'episodic', look_ahead = ['*'], historic_data = ['*'], back_orders = False,
                 rd_seed = 0, wd = True, file_name = True, **kwargs):


        seed(rd_seed)
        
        ### Main parameters ###
        self.M = 10                                     # Suppliers
        self.K = 10                                     # Products
        self.F = 4                                      # Fleete
        
        ### Other parameters ### 
        self.wh_cap = 1e9                               # Warehouse capacity
        self.min_sprice = 1;  self.max_sprice = 500
        self.min_hprice = 1;  self.max_hprice = 500
        self.penalization_cost = 1e9
        self.lambda1 = 0.5
        
        self.hor_typ = horizon_type == 'episodic'
        if self.hor_typ:    self.T = 6
        
        ### Look-ahead parameters ###
        if look_ahead:    
            self.S = 4              # Number of sample paths
            self.LA_horizon = 5     # Look-ahead time window's size
        
        ### Historic log parameters ###
        if type(historic_data) == list:        
            self.hist_window = 40       # Historic window

        ### Back-orders parameters ###
        if back_orders == 'back-orders':
            self.back_o_cost = 20
        elif back_orders == 'back-logs':
            self.back_l_cost = 20

        ### Custom configurations ###
        if file_name:
            utils.assign_env_config(self, kwargs)
            self.gen_sets()

        ### State space ###
        # Physical state
        self.state = {}     # Inventory
        
        ### Extra information ###
        self.others = {'look_ahead':look_ahead, 'historic': historic_data, 'wd': wd, 'file_name': file_name, 
                        'back_orders': back_orders}


    # Reseting the environment
    def reset(self, return_state = False):
        '''
        Reseting the environment. Genrate or upload the instance.
        PARAMETER:
        return_state: Indicates whether the state is returned
         
        '''   
        # Environment generated data
        if self.others['file_name']:  
            # General parameters
            self.gen_det_params()
            self.Ages = {k: range(1,self.O_k[k] + 1) for k in self.Products}
            self.gen_instance_data()
            
            # Episodic horizon
            if self.hor_typ:
                # Cuerrent time-step
                self.t = 0

            ## State ##
            self.state = {(k,o):0 for k in self.Products for o in self.Ages[k]}
            if self.others['back_orders'] == 'back-logs':
                for k in self.Products:
                    self.state[k,'B'] = 0

            if self.hor_typ:
                self.p = {(i,k): self.p_t[i,k,self.t] for i in self.Suppliers for k in self.Products}
                self.q = {(i,k): self.q_t[i,k,self.t] for i in self.Suppliers for k in self.Products}
                self.h = {k: self.h_t[k,self.t] for k in self.Products}
                self.d = {k: self.d_t[k,self.t] for k in self.Products}
            else:
                self.gen_realization()

            # Look-ahead, sample paths
            if self.others['look_ahead']:
                self.gen_sample_paths()                        
            
        # TODO! Data file upload 
        else:
            # Cuerrent time-step
            self.t = 0
            
            # Upload parameters from file
            self.O_k, self.c, self.Q, self.h_t, self.M_kt, self.K_it, self.q_t, self.p_t, 
            self.d_t, inventory = self.upload_instance(self.file_name, self.wd)
            
            # State
            self.state = inventory
        
        if return_state:    return self.state
        
    
    # Step 
    def step(self, action, validate_action = False, warnings = True):
        valid = True
        if validate_action:
            valid, error_msg = self.action_validity(action)

        if valid:
            # Inventory dynamics
            s_tprime, reward = self.transition_function(action, warnings)

            # Reward
            transport_cost, purchase_cost, holding_cost, back_orders_cost = self.compute_costs(action, s_tprime)
            reward += transport_cost + purchase_cost + holding_cost + back_orders_cost
    
            # Time step update and termination check
            self.t += 1
            done = self.check_termination(s_tprime)

            # State update
            if not done:
                self.update_state(s_tprime)
        
            # EXTRA INFORMATION TO BE RETURNED
            _ = {'p': self.p, 'q': self.q, 'h': self.h, 'd': self.d}
            if self.others['historic']:
                _['historic_info'] = self.historic_data
            if self.others['look_ahead']:
                _['sample_paths'] = self.sample_paths
            
            return self.state, reward, done, _
        
        else:
            print(colored('Time-step transition ERROR! The action is not valid. ' + error_msg, 'red'))
    
    
    def action_validity(self, action):
        routes, purchase, demand_complience = action[:3]
        if self.others['back_orders'] == 'back-logs':   back_o_complience = action[3]
        valid = True
        error_msg = ''
        
        # Route check
        if len(routes) > self.F:
            valid = False
            error_msg = 'The number of routes exceedes the number of vehicles'
            return valid, error_msg

        for route in routes:
            if route[0] != 0 or route[len(route) - 1] != 0:
                valid = False
                error_msg = 'Routes not valid, must start and end at the depot'
                return valid, error_msg
            for node in route:
                if node not in self.V:
                    valid = False
                    error_msg = 'Route must be made for nodes the set'
                    return valid, error_msg

        # Purchase
        for i in self.Suppliers:
            for k in self.Products:
                if purchase[i,k] > self.q[i,k]:
                    valid = False
                    error_msg = f"Purchased quantities exceed suppliers' available quantities  ({i},{k})"
                    return valid, error_msg
        
        # Demand_complience
        for k in self.Products:
            if self.others['back_orders'] != 'back-logs' and demand_complience[k,0] > sum(purchase[i,k] for i in self.Suppliers):
                valid = False
                error_msg = f'Demand complience with purchased of product {k}items exceed the purchase'
                return valid, error_msg
            elif self.others['back_orders'] == 'back-logs' and demand_complience[k,0] + back_o_complience[k,0] > sum(purchase[i,k] for i in self.Suppliers):
                valid = False
                error_msg = f'Demand/Back-logs complience with purchased items of product {k} exceed the purchase'
                return valid, error_msg

            if sum(demand_complience[k,o] for o in range(self.O_k[k] + 1)) > self.d[k]:
                valid = False
                error_msg = f'Trying to comply a non-existing demand of product {k}'
                return valid, error_msg
            
            for o in range(1, self.O_k[k] + 1):
                if self.others['back_orders'] != 'back-logs' and demand_complience[k,o] > self.state[k,o]:
                    valid = False
                    error_msg = f'Demand complience with inventory items exceed the stored items  ({k},{o})'
                    return valid, error_msg

                elif self.others['back_orders'] == 'back-logs' and demand_complience[k,o] + back_o_complience[k,o] > self.state[k,o]:
                    valid = False
                    error_msg = f'Demand/Back-logs complience with inventory items exceed the stored items ({k},{o})'
                    return valid, error_msg

        # Back-logs
        if self.others['back_orders'] == 'back-logs':
            for k in self.Products:
                if sum(back_o_complience[k,o] for o in range(self.O_k[k])) > self.state[k,'B']:
                    valid = False
                    error_msg = f'Trying to comply a non-existing back-log of product {k}'
                    return valid, error_msg
        
        elif self.others['back_orders'] == False:
            for k in self.Products:
                if sum(demand_complience[k,o] for o in range(self.O_k[k] + 1)) < self.d[k]:
                    valid = False
                    error_msg = f'Demand of product {k} was not fullfiled'
                    return valid, error_msg

        return valid, error_msg


    # Compute costs of a given procurement plan for a given day
    def compute_costs(self, action, s_tprime):
        routes, purchase, demand_complience = action[:3]
        if self.others['back_orders'] == 'back-logs':   back_o_complience = action[3]

        transport_cost = 0
        for route in routes:
            transport_cost += sum(self.c[route[i], route[i + 1]] for i in range(len(route) - 1))
        
        purchase_cost = sum(purchase[i,k] * self.p[i,k]   for i in self.Suppliers for k in self.Products)
        
        holding_cost = sum(sum(s_tprime[k,o] for o in range(1, self.O_k[k] + 1)) * self.h[k] for k in self.Products)

        back_orders_cost = 0
        if self.others['back_orders'] == 'back-orders':
            back_orders = round(sum(max(self.d[k] - sum(demand_complience[k,o] for o in range(self.O_k[k]+1)),0) for k in self.Products),1)
            print(f'Back-orders: {back_orders}')
            back_orders_cost = back_orders * self.back_o_cost
        
        elif self.others['back_orders'] == 'back-logs':
            back_orders_cost = sum(s_tprime[k,'B'] for k in self.Products) * self.back_l_cost

        return transport_cost, purchase_cost, holding_cost, back_orders_cost 
            
    
    # Inventory dynamics of the environment
    def transition_function(self, action, warnings):
        purchase, demand_complience = action[1:3]
        # Back-logs
        if self.others['back_orders'] == 'back-logs':   back_o_complience = action[3]
        inventory = deepcopy(self.state)
        reward  = 0

        # Inventory update
        for k in self.Products:
            inventory[k,1] = round(sum(purchase[i,k] for i in self.Suppliers) - demand_complience[k,0],1)

            max_age = self.O_k[k]
            if max_age > 1:
                for o in range(2, max_age + 1):
                        inventory[k,o] = round(self.state[k,o - 1] - demand_complience[k,o - 1],1)
            
            if self.others['back_orders'] == 'back-logs':
                new_back_logs = round(max(self.d[k] - sum(demand_complience[k,o] for o in range(self.O_k[k] + 1)),0),1)
                inventory[k,'B'] = round(self.state[k,'B'] + new_back_logs - sum(back_o_complience[k,o] for o in range(self.O_k[k]+1)),1)
                 

            # Factibility checks         
            if warnings:
                if self.state[k, max_age] - demand_complience[k,max_age] > 0:
                    reward += self.penalization_cost
                    print(colored(f'Warning! {self.state[k, max_age]} units of {k} were lost due to perishability','yellow'))

                if sum(demand_complience[k,o] for o in range(self.O_k[k] + 1)) < self.d[k]:
                    print(colored(f'Warning! Demand of product {k} was not fullfiled', 'yellow'))

            # if sum(inventory[k,o] for k in self.Products for o in range(self.O_k[k] + 1)) > self.wh_cap:
            #     reward += self.penalization_cost
            #     print(f'Warning! Capacity of the whareouse exceeded')

        return inventory, reward

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
        # Update historicals
        for k in self.Products:
            for i in self.Suppliers:
                if 'p' in self.others['historic']  or '*' in self.others['historic']:
                    self.historic_data['p'][i,k].append(self.p[i,k])
                if 'q' in self.others['historic']  or '*' in self.others['historic']:
                    self.historic_data['q'][i,k].append(self.q[i,k])
            if 'h' in self.others['historic']  or '*' in self.others['historic']:
                self.historic_data['h'][k].append(self.h[k])
            if 'd' in self.others['historic']  or '*' in self.others['historic']:
                self.historic_data['d'][k].append(self.d[k])

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
        pass
    

    # Auxiliary method: Generate iterables of sets
    def gen_sets(self):
    
        self.Suppliers = range(1,self.M);  self.V = range(self.M)
        self.Products = range(self.K)
        self.Vehicles = range(self.F)
        self.Samples = range(self.S)
        self.Horizon = range(self.T)
        self.TW = range(-self.hist_window, self.T)
        self.Historic = range(-self.hist_window, 0)
            

    # Generate deterministic parameters 
    def gen_det_params(self):
        ''' 
        Rolling horizon model parameters generation function
        Generates:
            - O_k: (dict) maximum days that k \in K can be held in inventory
            - c: (dict) transportation cost between nodes i \in V and j \in V
        '''
        # Maximum days that product k can be held in inventory before rotting
        self.O_k = {k:randint(1, self.T) for k in self.Products}
        
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid), randint(0, size_grid)) for i in self.V}
        # Transportation cost between nodes i and j, estimated using euclidean distance
        self.c = {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in self.V for j in self.V if i!=j} 
    
    
    # Auxiliary function to manage historic and simulated data 
    def gen_instance_data(self):
        if type(self.others['historic']) == list: 
            self.gen_simulated_data()
   
        elif type(self.others['historic']) == str:  
            self.upload_historic_data()
        
        else:
            raise ValueError('Historic information parameter value not valid')
                  
    
    # Generate historic and simulated stochastic parameters based on the requirement
    def gen_simulated_data(self):
        ''' 
        Simulated historical and sumulated data generator for quantities, prices and demand of products in each period.
        Generates:
            - h_t: (dict) holding cost of k \in K on t \in T
            - M_kt: (dict) subset of suppliers that offer k \in K on t \in T
            - K_it: (dict) subset of products offered by i \in M on t \in T
            - q_t: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            - p_t: (dict) price of k \in K offered by supplier i \in M on t \in T
            - d_t: (dict) demand of k \in K on t \in T
            - historic_data: (dict) with generated historic values
        '''
        self.historic_data = {}
        # Random holding cost of product k on t
        if 'h' in self.others['historic'] or  '*' in self.others['historic']:   
            self.historic_data['h'] = {k: [randint(self.min_hprice, self.max_hprice) for t in self.Historic] for k in self.Products}
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
        if 'q' in self.others['historic'] or  '*' in self.others['historic']:
            self.historic_data['q']= {(i,k): [randint(1,15) if i in self.M_kt[k,t] else 0 for t in self.Historic] for i in self.Suppliers for k in self.Products}
        self.q_t = {(i,k,t):randint(1,15) if i in self.M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products for t in self.Horizon}

        # Random price of available product k, provided by supplier i on t
        if 'p' in self.others['historic'] or  '*' in self.others['historic']:
            self.historic_data['p'] = {(i,k): [randint(1,500) if i in self.M_kt[k,t] else 1000 for t in self.Historic] for i in self.Suppliers for k in self.Products for t in self.Historic}
        self.p_t = {(i,k,t):randint(1,500) if i in self.M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products for t in self.Horizon}

        # Demand estimation based on quantities - ensuring feasibility, no backlogs
        if 'd' in self.others['historic'] or  '*' in self.others['historic']:
            self.historic_data['d'] = {(k):[(self.lambda1 * max([self.historic_data['q'][i,k][t] for i in self.Suppliers]) + (1-self.lambda1)*sum([self.historic_data['q'][i,k][t] for i in self.Suppliers])) for t in self.Historic] for k in self.Products}
        self.d_t = {(k,t):round((self.lambda1 * max([self.q_t[i,k,t] for i in self.Suppliers]) + (1-self.lambda1)*sum([self.q_t[i,k,t] for i in self.Suppliers])),1) for k in self.Products for t in self.Horizon}
    
   
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
            - hist_T: (int) number of periods that the historical datasets have information of
            - today: (int) current time period
        '''
        self.sample_paths = {}
        
        for s in self.Samples:
            # For each product, on each period chooses a random subset of suppliers that the product has had
            self.sample_paths[('M_k',s)] = {(k,t): [self.M_kt[k,tt] for tt in range(-self.hist_window + 1, self.t)][randint(-self.hist_window + 1, self.t - 1)] for k in self.Products for t in range(1, self.LA_horizon)}
            for k in self.Products:
                self.sample_paths[('M_k',s)][(k,0)] = self.M_kt[k, self.t]
            
            # Products offered by each supplier on each time period, based on M_kts
            self.sample_paths[('K_i',s)] = {(i,t): [k for k in self.Products if i in self.sample_paths[('M_k',s)][(k,t)]] \
                for i in self.Suppliers for t in range(1, self.LA_horizon)}
            for i in self.Suppliers:
               self.sample_paths[('K_i',s)][(k,0)] = self.K_it[i, self.t]
            

            # For each supplier and product, on each period chooses a quantity to offer using the sample value generator function
            #if 'q' in self.others['look_ahead']:
            self.sample_paths[('q',s)] = {(i,k,t): self.sim(self.historic_data['q'][i,k]) if i in self.sample_paths[('M_k',s)][(k,t)] else 0 \
                for i in self.Suppliers for k in self.Products for t in range(1, self.LA_horizon)}
            for i in self.Suppliers:
                for k in self.Products:
                    self.sample_paths[('q',s)][(i,k,0)] = self.q[i,k]
            
            # For each supplier and product, on each period chooses a price using the sample value generator function
            if 'p' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
                self.sample_paths[('p',s)] = {(i,k,t): self.sim(self.historic_data['p'][i,k]) if i in self.sample_paths[('M_k',s)][(k,t)] else 1000 \
                    for i in self.Suppliers for k in self.Products for t in range(1, self.LA_horizon)}
                for i in self.Suppliers:
                    for k in self.Products:
                        self.sample_paths[('p',s)][i,k,0] = self.p[i,k]
            
            if 'h' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
                self.sample_paths[('h',s)] = {(k,t): self.sim(self.historic_data['h'][k]) for k in self.Products for t in range(1, self.LA_horizon)}
                for k in self.Products:
                    self.sample_paths[('h',s)][k,0] = self.h[k]
            
            # Estimates demand for each product, on each period, based on q_s
            if 'd' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
                self.sample_paths[('d',s)] = {(k,t): (self.lambda1 * max([self.sample_paths[('q',s)][(i,k,t)] for i in self.Suppliers]) + (1 - self.lambda1) * sum([self.sample_paths[('q',s)][(i,k,t)] \
                    for i in  self.Suppliers])) for k in self.Products for t in range(1, self.LA_horizon)}
                for k in self.Products:
                    self.sample_paths[('d',s)][k,0] = self.d[k]
            
            # Vehicle capacity estimation
            # if 'Q' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
            #     self.sample_paths[('Q',s)] = 1.2 * self.gen_Q()
            
            # Set of vehicles, based on estimated required vehicles
            # if 'F' in self.others['look_ahead'] or '*' in self.others['look_ahead']:
            #     self.sample_paths[('F',s)] = int(sum(self.sample_paths[('d',s)].values())/self.sample_paths[('Q',s)]+1)


    # TODO! Generate a realization of random variables for continuous time-horizon
    def gen_realization(self):
        pass

    
    # Load parameters from a .txt file
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


    # Auxiliary method: Processing data from txt file 
    def extra_processing(self, coor):
        
        F = int(np.ceil(sum(self.d.values())/self.Q)); self.Vehicles = range(self.F)
        I_0 = {(k,o):0 for k in self.Products for o in range(1, self.O_k[k] + 1)} # Initial inventory level with an old > 1 
        
        # Travel cost between (i,j). It is the same for all time t
        c = {(i,j,t,v):round(np.sqrt(((coor[i][0]-coor[j][0])**2)+((coor[i][1]-coor[j][1])**2)),0) for v in self.Vehicles for t in self.Horizon for i in self.V for j in self.V if i!=j }
        
        return F, I_0, c


    # TODO! Uploading historic data  
    def upload_historic_data(self):  
        '''
        Method uploads information from file     
        
        '''
        file = self.others['historic']
        print('Method must be coded')

        '''
        self.h_t =
        self.q_t =
        self.p_t =
        self.d_t =
        '''


    def __repr__(self):
        return f'Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance. V = {self.M}; K = {self.K}; F = {self.F}'
    
        
        