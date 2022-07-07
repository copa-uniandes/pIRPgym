
"""
@author: juanbeta
    a
!TODO:
!!! SAMPLE PATHS
- Back-orders

- Instance_file uploading
- Historic_file uploading
- Renderizing
- MIP ADAPTATION

FUTURE WORK - Not completely developed:
- Continuous

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
        state: Current available inventory (!*):  (dict) - Inventory of k \in K of age o \in O_k
                                                           Age 'B' when back-logs
    - Other deterministic info (Z_t):
        p: Prices: (dict) Price of k \in K at i \in M
        q: Available quantities: (dict) Available quantities of k \in K at i \in M
        h: Holding cost: (dict) Holding cost of k \in K
        historic_data: (dict) Historic log of information (optional)
    - Belief State (B_t):
        sample_paths: Sample paths

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
    def __init__(self, horizon_type = 'episodic', look_ahead = ['d'], historic_data = ['d'], back_orders = False,
                 rd_seed = 0, wd = True, file_name = True, *args, **kwargs):


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
    def step(self, action, validate_action = False):
        valid = True
        if validate_action:
            valid, error_msg = self.action_validity(action)

        if valid:
            # Inventory dynamics
            s_tprime, reward = self.transition_function(action)

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
                    error_msg = "Purchased quantities exceed suppliers' available quantities"
                    return valid, error_msg
        
        # Demand_complience
        for k in self.Products:
            if self.others['back_orders'] != 'back-logs' and demand_complience[k,0] > sum(purchase[i,k] for i in self.Suppliers):
                valid = False
                error_msg = 'Demand complience with purchased items exceed the purchase'
                return valid, error_msg
            elif self.others['back_orders'] == 'back-logs' and demand_complience[k,0] + back_o_complience[k,0] > sum(purchase[i,k] for i in self.Suppliers):
                valid = False
                error_msg = 'Demand/Back-logs complience with purchased items exceed the purchase'
                return valid, error_msg

            if sum(demand_complience[k,o] for o in range(self.O_k[k] + 1)) > self.d[k]:
                valid = False
                error_msg = 'Trying to comply a non-existing demand'
                return valid, error_msg
            
            for o in range(1, self.O_k[k] + 1):
                if self.others['back_orders'] != 'back-logs' and demand_complience[k,o] > self.state[k,o]:
                    valid = False
                    error_msg = 'Demand complience with inventory items exceed the stored items'
                    return valid, error_msg

                elif self.others['back_orders'] == 'back-logs' and demand_complience[k,o] + back_o_complience[k,o] > self.state[k,o]:
                    valid = False
                    error_msg = 'Demand/Back-logs complience with inventory items exceed the stored items'
                    return valid, error_msg
        # Back-logs
        if self.others['back_orders'] == 'back-logs':
            for k in self.Products:
                if sum(back_o_complience[k,o] for o in self.Ages[k]) < self.state[k,'B']:
                    valid = False
                    error_msg = 'Trying to comply a non-existing back-log'
                    return valid, error_msg
        
        elif self.others['back_orders'] == False:
            if sum(demand_complience[k,o] for o in range(self.O_k[k] + 1)) < self.d[k]:
                valid = False
                error_msg = 'Demand was not fullfiled'
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
            back_orders_cost = sum(max(self.d[k] - sum(demand_complience[k,o] for o in self.Ages[k]),0) * self.back_o_cost for k in self.Products) \
                * self.back_o_cost
        
        elif self.others['back_orders'] == 'back-logs':
            back_orders_cost = sum(s_tprime[k,'B'] for k in self.Products) * self.back_l_cost


        return transport_cost, purchase_cost, holding_cost, back_orders_cost 
            
    
    # Inventory dynamics of the environment
    def transition_function(self, action):
        purchase, demand_complience = action[1:3]
        # Back-logs
        if self.others['back_orders'] == 'back-logs':   back_o_complience = action[3]
        inventory = deepcopy(self.state)
        reward  = 0

        # Inventory update
        for k in self.Products:
            inventory[k,1] = sum(purchase[i,k] for i in self.Suppliers) - demand_complience[k,0]

            max_age = self.O_k[k]
            if max_age > 1:
                for o in range(2, max_age + 1):
                        inventory[k,o] = self.state[k,o - 1] - demand_complience[k,o - 1]

            # Factibility checks         
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
        self.d_t = {(k,t):(self.lambda1 * max([self.q_t[i,k,t] for i in self.Suppliers]) + (1-self.lambda1)*sum([self.q_t[i,k,t] for i in self.Suppliers])) for k in self.Products for t in self.Horizon}
    
   
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


    # # Generate vehicle's capacity
    # def gen_Q(self):
    #     """
    #     Vehicle capacity parameter generator
    #     Returns feasible vehicle capacity to use in rolling horizon model. 
    #     """ 
    #     m = gu.Model("Q")
        
    #     z = {(i,k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"z_{i,k,t}") for k in self.Products for t in range(self.LA_horizon) for i in self.M_kt[k,t]}
    #     Q = m.addVar(vtype=gu.GRB.CONTINUOUS)
        
    #     for k in self.Products:
    #         for t in range(self.LA_horizon):
    #             # Demand constraint - must buy from suppliers same demanded quantity
    #             m.addConstr(gu.quicksum(z[i,k,t] for i in self.M_kt[k,t]) == self.d_t[k,t])
            
    #     for t in range(self.LA_horizon):
    #         for i in self.Suppliers:
    #             m.addConstr(gu.quicksum(z[i,k,t] for k in self.Products if (i,k,t) in z) <= Q)
        
    #     m.setObjective(Q);  m.update();   m.setParam("OutputFlag",0)
    #     m.optimize()
        
    #     if m.Status == 2:
    #         return Q.X
    #     else:
    #         print("Infeasible")
    #         return 0 

    
    # # Solution Approach: MIP embedded on a rolling horizon
    # def MIP_rolling_horizon(self, verbose = False):
        
    #     start = time.time()

    #     today = self.Historical - self.T - 1
    #     self.target = range(today, today + self.T)
    #     # Days to simulate (including today)
    #     self.base = today + 0.

    #     self.res = {}
    #     for tau in self.target:
            
    #         # Time window adjustment
    #         adj_horizon = min(self.horizon_T, self.Historical - today - 1)
    #         T = range(adj_horizon)
            
    #         q_s, p_s, d_s, M_kts, K_its, Q_s, F_s = self.gen_sim_paths(today)
            
    #         for s in range(len(Q_s)):
    #             Q_s[s] *= 5
    #             F_s[s] = range(int(len(F_s[s])/4))
    #         F_s = {s:range(max([len(F_s[ss]) for ss in self.Samples])) for s in self.Samples}
            
    #         m = gu.Model('TPP+IM_perishable')
            
    #         ''' Decision Variables '''
            
    #         ###### TPP variables
            
    #         # If vehicle v \in F uses arc (i,j) on T, for vsample path s \in Sam
    #         x = {(i,j,t,v,s):m.addVar(vtype=gu.GRB.BINARY, name=f"x_{i,j,t,v,s}") for s in self.Samples for i in self.V for j in self.V for t in T for v in F_s[s] if i!=j}
    #         # If supplier i \in M is visited by vehicle v \in F on t \in T, for sample path s \in Sam
    #         w = {(i,t,v,s):m.addVar(vtype=gu.GRB.BINARY, name=f"w_{i,t,v,s}") for s in self.Samples for i in self.Suppliers for t in T for v in F_s[s]}
    #         # Quantity of product k \in K bought from supplier i \in M on t \in T, transported in vehicle v \in F, for sample path s \in Sam
    #         z = {(i,k,t,v,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"z_{i,k,t,v,s}") for s in self.Samples for k in self.Products for t in T for i in M_kts[s][k,t] for v in F_s[s]}
    #         # Auxiliar variable for routing modeling
    #         u = {(i,t,v,s):m.addVar(vtype=gu.GRB.BINARY, name=f"u_{i,t,v,s}") for s in self.Samples for i in self.Suppliers for t in T for v in F_s[s]}
            
    #         ###### Inventory variables
            
    #         # Quantity of product k \in K aged o \in O_k shipped to final customer on t, for sample path s \in Sam
    #         y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"y_{k,t,o,s}") for k in self.Products for t in T for o in range(1, self.O_k[k]+1) for s in self.Samples}
            
    #         ''' State Variables '''
            
    #         # Quantity of product k \in K that is replenished on t \in T, for sample path s \in Sam
    #         r = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"r_{k,t,s}") for k in self.Products for t in T for s in self.Samples}
    #         # Inventory level of product k \in K aged o \in O_k on t \in T, for sample path s \in Sam
    #         ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"i_{k,t,o,s}") for k in self.Products for t in T for o in range(1, self.O_k[k]+1) for s in self.Samples}
            
    #         ''' Constraints '''
    #         for s in self.Samples:
    #             #Inventory constraints
    #             ''' For each product k aged o, the inventory level on the first time period is
    #             the initial inventory level minus what was shipped of it in that same period'''
    #             for k in self.Products:
    #                 for o in range(1, self.O_k[k]+1):
    #                     if o > 1:
    #                         m.addConstr(ii[k,0,o,s] == self.I_0[k,o]-y[k,0,o,s])
                
    #             ''' For each product k on t, its inventory level aged 0 is what was replenished on t (r)
    #             minus what was shipped (y)'''
    #             for k in self.Products:
    #                 for t in T:
    #                     m.addConstr(ii[k,t,1,s] == r[k,t,s]-y[k,t,1,s])
                        
    #             ''' For each product k, aged o, on t, its inventory level is the inventory level
    #             aged o-1 on the previous t, minus the amount of it that's shipped '''
    #             for k in self.Products:
    #                 for t in T:
    #                     for o in range(1, self.O_k[k]+1):
    #                         if t>0 and o>1:
    #                             m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s]-y[k,t,o,s])                
                
    #             ''' The amount of product k that is shipped to the final customer on t is equal to its demand'''
    #             for k in self.Products:
    #                 for t in T:
    #                     m.addConstr(gu.quicksum(y[k,t,o,s] for o in range(1, self.O_k[k]+1)) == d_s[s][k,t])
                        
    #             #TPP constrains
    #             ''' Modeling of state variable r_kt. For each product k, on each t, its replenishment 
    #             is the sum of what was bought from all the suppliers'''
    #             for t in T:
    #                 for k in self.Products:
    #                     m.addConstr(gu.quicksum(z[i,k,t,v,s] for i in M_kts[s][k,t] for v in F_s[s]) == r[k,t,s])
                        
    #             ''' Cannot buy from supplier i more than their available quantity of product k on t '''
    #             for t in T:
    #                 for k in self.Products:
    #                     for i in M_kts[s][k,t]:
    #                         m.addConstr(gu.quicksum(z[i,k,t,v,s] for v in F_s[s]) <= q_s[s][i,k,t])
                            
    #             ''' Can only buy product k from supplier i on t and pack it in vehicle v IF
    #              vehicle v is visiting the supplier on t'''        
    #             for t in T:
    #                 for v in F_s[s]:
    #                     for k in self.Products:
    #                         for i in M_kts[s][k,t]:
    #                             m.addConstr(z[i,k,t,v,s] <= q_s[s][i,k,t]*w[i,t,v,s])
                
    #             ''' At most, one vehicle visits each supplier each day'''
    #             for t in T:
    #                 for i in self.Suppliers:
    #                     m.addConstr(gu.quicksum(w[i,t,v,s] for v in F_s[s]) <= 1)
                
    #             ''' Must respect vehicle's capacity '''
    #             for t in T:
    #                 for v in F_s[s]:
    #                     m.addConstr(gu.quicksum(z[i,k,t,v,s] for k in self.Products for i in M_kts[s][k,t] ) <= Q_s[s])
                
    #             ''' Modeling of variable w_itv: if supplier i is visited by vehicle v on t'''
    #             for v in F_s[s]:
    #                 for t in T:
    #                     for hh in self.Suppliers:
    #                         m.addConstr(gu.quicksum(x[hh,j,t,v,s] for j in self.V if hh!=j) ==  w[hh,t,v,s])
    #                         m.addConstr(gu.quicksum(x[i,hh,t,v,s] for i in self.V if i!=hh) ==  w[hh,t,v,s])
                
    #             ''' Routing modeling - no subtours created'''          
    #             for t in T:
    #                 for i in self.Suppliers:
    #                     for v in F_s[s]:
    #                         for j in self.Suppliers:
    #                             if i!=j:
    #                                 m.addConstr(u[i,t,v,s] - u[j,t,v,s] + len(self.V) * x[i,j,t,v,s] <= len(self.V)-1 )
            
    #             ''' Non-Anticipativity Constraints'''
    #             for v in F_s[s]:
    #                 for i in self.V:
    #                     for j in self.V:
    #                         if i!=j:
    #                             m.addConstr(x[i,j,0,v,s] == gu.quicksum(x[i,j,0,v,s1] for s1 in self.Samples)/self.S)
                
    #             for v in F_s[s]:
    #                 for i in self.Suppliers:
    #                     m.addConstr(w[i,0,v,s] == gu.quicksum(w[i,0,v,s1] for s1 in self.Samples)/self.S)
                
    #             for v in F_s[s]:
    #                 for k in self.Products:
    #                     for i in M_kts[s][k,0]:
    #                         m.addConstr(z[i,k,0,v,s] == gu.quicksum(z[i,k,0,v,s1] for s1 in self.Samples)/self.S)
                
    #             for i in self.Suppliers:
    #                 for v in F_s[s]:
    #                     m.addConstr(u[i,0,v,s] == gu.quicksum(u[i,0,v,s1] for s1 in self.Samples)/self.S)
                
    #             for k in self.Products:
    #                 for o in range(1, self.O_k[k]+1):
    #                     m.addConstr(y[k,0,o,s] == gu.quicksum(y[k,0,o,s1] for s1 in self.Samples)/self.S)
            
            
    #         ''' Costs '''
    #         routes = gu.quicksum(self.c[i,j] * x[i,j,t,v,s] for s in self.Samples for i in self.V for j in self.V for t in T for v in F_s[s] if i!=j)
    #         acquisition = gu.quicksum(p_s[s][i,k,t]*z[i,k,t,v,s] for s in self.Samples for k in self.Products for t in T for v in F_s[s] for i in M_kts[s][k,t])
    #         holding = gu.quicksum(self.h[k,t]*ii[k,t,o,s] for s in self.Samples for k in self.Products for t in T for o in range(1, self.O_k[k]+1))
                            
    #         m.setObjective((routes+acquisition+holding)/self.S)
            
    #         m.update()
    #         m.setParam("OutputFlag", verbose)
    #         m.setParam("MIPGap",0.01)
    #         m.setParam("TimeLimit", 3600)
    #         m.optimize()
           
    #         self.tiempoOpti = time.time() - start
    #         print(f"Done {tau}")
            
            
    #         ''' Initial inventory level updated for next iteration '''
    #         self.I_0 = {(k,o):ii[k,0,o-1,0].X if o > 1 else 0 for k in self.Products for o in range(1, self.O_k[k]+1)}
            
    #         ''' Saves results for dashboard '''
    #         self.ii = {(k,o):ii[k,0,o,0].X for k in self.Products for o in range(1, self.O_k[k]+1)}
    #         self.r = {k:r[k,0,0].X for k in self.Products}
    #         self.z = {(i,k,v):z[i,k,0,v,0].X for k in self.Products for i in M_kts[0][k,0] for v in F_s[0]}
    #         self.x = {(i,j,v):x[i,j,0,v,0].X for i in self.V for j in self.V for v in F_s[0] if i!=j}
    #         self.p_s = {(i,k):sum(p_s[s][i,k,0] for s in self.Samples)/self.S for i in self.Suppliers for k in self.Products}
    #         self.res[tau] = (ii,r,x,d_s,F_s[0],adj_horizon,p_s,z)
            
    #         today += 1
        
    #     print(self.tiempoOpti)
    
    # def result_renderization(self, path, save = True, show = True):
        
    #     images = []
    #     ''' Rolling Horizon visualization'''

    #     M_kt, K_it, q, p, d = self.Historic_log

    #     def unique(lista):
    #         li = []
    #         for i in lista:
    #             if i not in li:
    #                 li.append(i)
    #         return li

    #     for tau in self.target:
    #         cols = {0:"darkviolet",1:"darkorange",2:"lightcoral",3:"seagreen", 4:"dimgrey", 5:"teal", 6:"olive", 7:"darkmagenta"}
    #         fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,figsize=(13,20))
    #         T = range(self.res[tau][5])
            
    #         ''' Demand Time Series and Sample Paths '''
    #         for s in self.Samples:
    #             ax1.plot([tt for tt in range(tau - self.Historical + self.T + 1, 
    #                                          tau - self.Historical + self.T + 1 + self.res[tau][5])],
    #                      [sum(self.res[tau][3][s][k,t] for k in self.Products) for t in T],color=cols[s])
    #         for tt in range(tau-self.Historical+self.T+2):
    #             ax1.plot(tt,sum(d[k, self.base+tt] for k in self.Products),"*",color="black",markersize=12)
    #         ax1.plot([tt for tt in range(tau - self.Historical + self.T + 2)],
    #                  [sum(d[k, self.base+tt] for k in self.Products) for tt in range(tau-self.Historical+self.T+2)],
    #                  color="black")
    #         ax1.set_xlabel("Time period")
    #         ax1.set_ylabel("Total Demand")
    #         ax1.set_xticks(ticks=[t for t in range(self.T)])
    #         ax1.set_xlim(-0.5,self.T-0.5)
    #         ax1.set_ylim(min([sum(d[k,t] for k in self.Products) for t in range(self.Historical)])-20,max([sum(d[k,t] for k in self.Products) for t in range(self.Historical)])+20)
            
    #         ''' Total Cost Bars '''
    #         xx = [tt for tt in range(tau-self.Historical+self.T+2)]
    #         compra = [sum(self.res[tt+self.Historical-self.T-1][6][i,k]*self.res[tt+self.Historical-self.T-1][7][i,k,v] for i in self.Suppliers for k in self.Products for v in self.res[tt+self.Historical-self.T-1][4] if (i,k,v) in self.res[tt+self.Historical-self.T-1][7]) for tt in xx]
    #         invent = [sum(self.h[k,tt]*self.res[tt+self.Historical-self.T-1][0][k,o] for k in self.Products for o in range(1, self.O_k[k]+1)) for tt in xx]
    #         rutas = [sum(self.c[i,j]*self.res[tt+self.Historical-self.T-1][2][i,j,v] for i in self.V for j in self.V for v in self.res[tt+self.Historical-self.T-1][4] if i!=j) for tt in xx]
    #         ax2.bar(x=xx,height=rutas,bottom=[compra[tt]+invent[tt] for tt in xx],label="Routing",color="sandybrown")
    #         ax2.bar(x=xx,height=invent,bottom=compra,label="Holding",color="slateblue")
    #         ax2.bar(x=xx,height=compra,label="Purchase",color="lightseagreen")
    #         ax2.set_xlabel("Time period")
    #         ax2.set_ylabel("Total Cost")
    #         ax2.set_xticks(ticks=[t for t in range(self.T)])
    #         ax2.set_xlim(-0.5,self.T-0.5)
    #         a1 = [sum(self.res[tt+self.Historical-self.T-1][6][i,k]*self.res[tt+self.Historical-self.T-1][7][i,k,v] for i in self.Suppliers for k in self.Products for v in self.res[tt+self.Historical-self.T-1][4] if (i,k,v) in self.res[tt+self.Historical-self.T-1][7]) for tt in range(self.T)]
    #         a2 = [sum(self.h[k,tt]*self.res[tt+self.Historical-self.T-1][0][k,o] for k in self.Products for o in range(1,self.O_k[k]+1)) for tt in range(self.T)]
    #         a3 = [sum(self.c[i,j]*self.res[tt+self.Historical-self.T-1][2][i,j,v] for i in self.V for j in self.V for v in self.res[tt+self.Historical-self.T-1][4] if i!=j) for tt in range(self.T)]
    #         lim_cost = max(a1[tt]+a2[tt]+a3[tt] for tt in range(self.T))
    #         ax2.set_ylim(0,lim_cost+5e3)
    #         ticks = [i for i in range(0,int(int(lim_cost/1e4+1)*1e4),int(1e4))]
    #         ax2.set_yticklabels(["${:,.0f}".format(int(i)) for i in ticks])
    #         ax2.legend(loc="upper right")
            
    #         ''' Inventory Level by Product '''
    #         cols_t = {0:(224/255,98/255,88/255),1:(224/255,191/255,76/255),2:(195/255,76/255,224/255),3:(54/255,224/255,129/255),4:(65/255,80/255,224/255),5:(224/255,149/255,103/255),6:(131/255,224/255,114/255),7:(224/255,103/255,91/255),8:(70/255,223/255,224/255),9:(224/255,81/255,183/255)}
    #         for tt in range(tau-self.Historical+self.T+2):
    #             list_inv = [round(sum(self.res[tt+self.Historical-self.T-1][0][k,o] for o in range(1,self.O_k[k]+1))) for k in self.Products]
    #             un_list = unique(list_inv)
    #             ax3.scatter(x=[tt for i in un_list],y=[i for i in un_list],s=[1e2+(1e4-1e2)*(sum([1 for j in list_inv if j==i])-1)/(self.K-1) for i in un_list],alpha=0.5,color=cols_t[tt])
    #         ax3.set_xticks(ticks=[t for t in range(self.T)])
    #         ax3.set_xlabel("Time period")
    #         ax3.set_ylabel("Inventory Level by product")
    #         ax3.set_xlim(-0.5,self.T-0.5)
    #         lim_inv = max([round(sum(self.res[tt+self.Historical-self.T-1][0][k,o] for o in range(1,self.O_k[k]+1))) for k in self.Products for tt in range(int(self.T))])
    #         ax3.set_ylim(0,lim_inv+1)
            
    #         ''' Purchased Quantity, by Product '''
    #         b_purc = []
    #         for tt in range(tau-self.Historical+self.T+2):
    #             aver = [round(self.res[tt+self.Historical-self.T-1][1][k]) for k in self.Products]
    #             b_purc.append(ax4.boxplot(aver,positions=[tt],widths=[0.5],patch_artist=True,flierprops={"marker":'o', "markerfacecolor":cols_t[tt],"markeredgecolor":cols_t[tt]}))
    #             for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    #                 if item != 'medians':
    #                     plt.setp(b_purc[-1][item], color=cols_t[tt])
    #                 else:
    #                     plt.setp(b_purc[-1][item], color="white")
    #         ax4.set_xlim(-0.5,self.T-0.5)
    #         ax4.set_xticks(ticks=[t for t in range(self.T)])
    #         ax4.set_xlabel("Time period")
    #         ax4.set_ylabel("Purchased Quantity, by product")
    #         lim_purc = max([round(self.res[tt+self.Historical-self.T-1][1][k]) for k in self.Products for tt in range(int(self.T))])
    #         ax4.set_ylim(0,lim_purc+5)
            
    #         plt.savefig(path+f"RH_d{tau}.png", dpi=300)
    #         if show:
    #             plt.show()
            
    #         if save:
    #             images.append(imageio.imread(path+f"RH_d{tau}.png"))
        
    #     if save:
    #         imageio.mimsave(path+"RH1.gif", images,duration=0.5)
    
    def __repr__(self):

        return f'Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance. V = {self.M}; K = {self.K}; F = {self.F}'
    
        
        