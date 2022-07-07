#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:15:27 2022

@author: juanbeta
"""

#%% Modules
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time; from random import random, seed, randint, shuffle

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

sys.path.insert(1, '/Users/juanbeta/Dropbox/Investigación/IB-MP-IRP-PP/Environment/')
import utils

### Gym & OR-Gym
import gym; from gym import spaces; 

#%% Description
'''
State (S): The state has, according to Powell, three main components:
    - Physical State (R_t):
        Current inventory
    - Other deterministic info (Z_t):
        Historic log of information
        Prices (*)
        Available quantities (*)
    - Belief State (B_t):
        Sample paths

Actions (X): The actions performed by the agent will be three
    Suppliers to visit
    Route to visit suppliers
    Quantity to purchase

Exogenous information (W): The stochastic factors considered on the environment
    Demand (*)
        
(*) Varying the stochastic factors might be of interest. Therefore, the deterministic factors
    will be under Z_t and stochastic factors will be generated and presented in the W_t
'''

#%% Steroid IRP class

class steroid_IRP(gym.Env):
    
    # Initialization method
    def __init__(self, rd_seed = 0, wd = True, file_name = True, *args, **kwargs):
        
        '''
        Initialization: Two initialization options
        1. !!! Upload instance from x.txt file in y directory
            - steroid_IRP(wd = y, file_name = x.txt)
        2. Generate instance with historical data for rolling-horizon
            - env_config = {'M': M, 'K': K,...}    (custom configurations)
            - steroid_IRP(env_config = env_config)
            
        Parameters
        rd_seed = 0: Seed for random number generation
        wd = True: Working directory path
        file_name = True: File name when uploading instances from .txt
        **kwargs: 
            M = 10: Number of suppliers
            K = 10: Number of Products
            T = 6:  Number of decision periods
            F = 2:  Number of vehicles on the fleete
            
            wh_cap = 1e9: Warehouse capacity
            min/max_sprice: Max and min selling prices (per m and k)
            min/max_hprice: Max and min holding cost (per k)
            
            S = 4:  Number of sample paths 
            LA_horizon = 5: Number of look-ahead periods
            lambda1 = 0.5: Controls demand, assures feasibility
        '''
        
        seed(rd_seed)
        
        ### Parameters ###
        self.M = 10             # Suppliers
        self.K = 10             # Products
        self.T = 6              # Time periods
        self.F = 2              # Fleete
        
        ### Extra parameters ###
        self.wh_cap = 1e9       # Warehouse capacity
        self.min_sprice = 1;  self.max_sprice = 500
        self.min_hprice = 1;  self.max_hprice = 500
        self
        
        ### Sample path generation parameters ###
        self.S = 4;   
        self.LA_horizon = 5
             
        self.lambda1 = 0.5      # Controls demand, assures feasibility
        self.replica = 2        
        self.Historical = 40 + self.T
        
        ### Initialization methods
        self.init_type = file_name == True  # 1: generated, 0: uploaded
        # TODO! Verify
        if not wd:   sys.path.insert(1, wd)
        # Generate instance with historical data 
        if file_name:  
            utils.assign_env_config(self, kwargs)       # Apply custom configurations
            self.generate_sets()
            self.generate_SP_instance()                 # Generate the historical data 
            
        else:   
            self.generate_sets()
            self.upload_params(file_name)
        
        self.h, self.O_k, self.c = self.gen_det_params()
    
        
        ### State space ###
        # Physical state
        self.obs_dim = self.K
        state_low = np.zeros(self.obs_dim)
        state_max = np.repeat(self.wh_cap, self.K)
        self.observation_space = spaces.Box(state_low, state_max, dtype = np.int32)
        
        # Other information 
        sprices_min = np.repeat(self.min_sprice, self.K * (self.M - 1))
        sprices_max = np.repeat(self.max_sprice, self.K * (self.M - 1))
        self.ext_info_st = spaces.Box(sprices_min, sprices_max, np.int32)
        
        hprices_min = np.repeat(self.min_hprice, self.K)
        hprices_max = np.repeat(self.max_hprice, self.K)
        self.ext_info_st = spaces.Box(hprices_min, hprices_max, np.int32) 
        
        # TODO! model historic information
        
        # Belief variables
        self.belief_st = {}   # TODO! model sample paths

        ### Action space ###
        self.act_dim = self.F * self.M + self.F * self.M * self.K
        action_low = np.zeros(self.F * self.M + self.F * self.M * self.K)
        action_high = np.repeat(self.K, self.K * self.F) + np.repeat(1e9, self.F * self.M * self.K)
        self.action_space = spaces.Box(action_low, action_high, dtype = np.int32)

        
    # Reseting the environment
    def reset(self, return_state = False):
        
        # General parameters
        self.t = 0
        self.generate_sets()
        
        # (Physical) State
        self.state = {(k,o):0 for k in self.Products for o in range(1, O_k[k]+1)}
        
        
        if not return_state:
            
            return self.state
        
        
        
        pass        # TODO!
    
    # Auxiliary method: Generate iterables of sets
    def generate_sets(self):
        
        self.Suppliers = range(1,self.M);  self.V = range(self.M)
        self.Products = range(self.K)
        self.Horizon = range(self.T)
        self.Vehicles = range(self.F)
        self.Samples = range(self.S)
    
    # Step 
    def step(self):
        pass        # TODO!
    
    # Instance generation with sample paths
    def generate_SP_instance(self):
        
        # Historic values: [M_kt, K_it, q, p, d]
        self.Historic_log = self.gen_historic_simulation()
        
        
    # Generate deterministic parameters 
    def gen_det_params(self):
        
        ''' 
        Rolling horizon model parameters generation function
        Returns:
            - h: (dict) holding cost of k \in K on t \in T
            - O_k: (dict) maximum days that k \in K can be held in inventory
            - c: (dict) transportation cost between nodes i \in V and j \in V
        '''
        # Random holding cost of product k on t
        h = {(k,t):randint(self.min_hprice, self.max_hprice) for k in self.Products for t in self.Horizon}
    
        # Maximum days that product k can be held in inventory before rotting
        O_k = {k:randint(1, self.T) for k in self.Products}
        
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid), randint(0, size_grid)) for i in self.V}
        # Transportation cost between nodes i and j, estimated using euclidean distance
        c = {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in self.V for j in self.V if i!=j}
        
        return h, O_k, c
 
    
    # Generate historic stochastic parameters
    def gen_historic_simulation(self):
        ''' 
        Simulated historical data generator for quantities, prices and demand of products in each period.
        Returns list with:
            - M_kt: (dict) subset of suppliers that offer k \in K on t \in T
            - K_it: (dict) subset of products offered by i \in M on t \in T
            - q: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            - p: (dict) price of k \in K offered by supplier i \in M on t \in T
            - d: (dict) demand of k \in K on t \in T
        '''
        
        TW = range(self.Historical)
        
        M_kt = {}
        # In each time period, for each product
        for k in self.Products:
            for t in TW:
                # Random number of suppliers that offer k in t
                sup = randint(1, self.M)
                M_kt[k,t] = list(self.Suppliers)
                # Random suppliers are removed from subset, regarding {sup}
                for ss in range(self.M - sup):
                    a = int(randint(0, len(M_kt[k,t])-1))
                    del M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        K_it = {(i,t):[k for k in self.Products if i in M_kt[k,t]] for i in self.Suppliers for t in TW}
        
        # Random quantity of available product k, provided by supplier i on t
        q = {(i,k,t):randint(1,15) if i in M_kt[k,t] else 0 for i in self.Suppliers for k in self.Products for t in TW}
        # Random price of available product k, provided by supplier i on t
        p = {(i,k,t):randint(1,500) if i in M_kt[k,t] else 1000 for i in self.Suppliers for k in self.Products for t in TW}
        
        # Demand estimation based on quantities - ensuring feasibility, no backlogs
        d = {(k,t):(self.lambda1 * max([q[i,k,t] for i in self.Suppliers]) + (1-self.lambda1)*sum([q[i,k,t] for i in self.Suppliers])) for k in self.Products for t in TW}
        
        return [M_kt, K_it, q, p, d]
    
    
    # Generate vehicle's capacity
    def gen_Q(self):
        ''' 
        Vehicle capacity parameter generator
        Returns feasible vehicle capacity to use in rolling horizon model. 
        '''
        M_kt, d  = self.Historic_log[::4] 
        m = gu.Model("Q")
        
        z = {(i,k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"z_{i,k,t}") for k in self.Products for t in range(self.horizon_T) for i in M_kt[k,t]}
        Q = m.addVar(vtype=gu.GRB.CONTINUOUS)
        
        for k in self.Products:
            for t in range(self.horizon_T):
                # Demand constraint - must buy from suppliers same demanded quantity
                m.addConstr(gu.quicksum(z[i,k,t] for i in M_kt[k,t]) == d[k,t])
            
        for t in range(self.horizon_T):
            for i in self.Suppliers:
                m.addConstr(gu.quicksum(z[i,k,t] for k in self.Products if (i,k,t) in z) <= Q)
        
        m.setObjective(Q);  m.update();   m.setParam("OutputFlag",0)
        m.optimize()
        
        if m.Status == 2:
            return Q.X
        else:
            print("Infeasible")
            return 0
        
        return Q.x
    
    
    # Sample value generator function
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
    def gen_sim_paths(self, today):
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
        M_kt, K_it, q, p, d = self.Historic_log
        
        # Day 0 is deterministic, so the horizon is the original value - 1
        hist_T = copy(today)
        hist_T -= 1
        
        M_kts, K_its = {},{}
        q_s,p_s,dem_s,Q_s,F_s = {},{},{},{},{}
        
        for s in self.Samples:
            # For each product, on each period chooses a random subset of suppliers that the product has had
            M_kts[s] = {(k,t):[M_kt[k,t] for tt in range(hist_T)][randint(0,hist_T-1)] for k in self.Products for t in range(1, self.horizon_T)}
            for k in self.Products:
                M_kts[s][k,0] = M_kt[k,today]
            
            # Products offered by each supplier on each time period, based on M_kts
            K_its[s] = {(i,t):[k for k in self.Products if i in M_kts[s][k,t]] for i in self.Suppliers for t in range(1, self.horizon_T)}
            for i in self.Suppliers:
                K_its[s][i,0] = K_it[i,today]
            
            # For each supplier and product, on each period chooses a quantity to offer using the sample value generator function
            q_s[s] = {(i,k,t):self.sim([q[i,k,tt] for tt in range(hist_T) if q[i,k,tt] > 0] ) if i in M_kts[s][k,t] else 0 for i in self.Suppliers for k in self.Products for t in range(1, self.horizon_T)}
            for i in self.Suppliers:
                for k in self.Products:
                    q_s[s][i,k,0] = q[i,k,today]
            
            # For each supplier and product, on each period chooses a price using the sample value generator function
            p_s[s] = {(i,k,t):self.sim([p[i,k,tt] for tt in range(hist_T) if p[i,k,tt]<1000]) if i in M_kts[s][k,t] else 1000 for i in  self.Suppliers for k in self.Products for t in range(1, self.horizon_T)}
            for i in self.Suppliers:
                for k in self.Products:
                    p_s[s][i,k,0] = p[i,k,today]
            
            # Estimates demand for each product, on each period, based on q_s
            dem_s[s] = {(k,t):(self.lambda1 * max([q_s[s][i,k,t] for i in self.Suppliers]) + (1 - self.lambda1)*sum([q_s[s][i,k,t] for i in  self.Suppliers])) for k in self.Products for t in range(1, self.horizon_T)}
            for k in self.Products:
                dem_s[s][k,0] = d[k,today]
            
            # Vehicle capacity estimation
            Q_s[s] = 1.2 * self.gen_Q()
            
            # Set of vehicles, based on estimated required vehicles
            F_s[s] = range(int(sum(dem_s[s].values())/Q_s[s])+1)
            
        return q_s, p_s, dem_s, M_kts, K_its, Q_s, F_s
    
    
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
        
        self.Mk = Mk;  self.Km = Km
        
        self.Q = Q;  self.O_k = O_k
        self.h = h ;  self.d = d;  self.q = q;  self.p = p
        
        # self.F, self.I_0, self.c, self.max_cij = self.extra_processing(coor)
        self.F, self.I_0, self.c  = self.extra_processing(coor)

    # Auxiliary method: Processing data from txt file (NOT FOR ROLLING-HORIZON)
    def extra_processing(self, coor):
        
        F = int(np.ceil(sum(self.d.values())/self.Q)); self.Vehicles = range(self.F)
        I_0 = {(k,o):0 for k in self.Products for o in range(1, self.O_k[k]+1)} # Initial inventory level with an old > 1 
        
        # Travel cost between (i,j). It is the same for all time t
        c = {(i,j,t,v):round(np.sqrt(((coor[i][0]-coor[j][0])**2)+((coor[i][1]-coor[j][1])**2)),0) for v in self.Vehicles for t in self.Horizon for i in self.V for j in self.V if i!=j }
        
        return F, I_0, c

    
    # Solution Approach: MIP embedded on a rolling horizon
    def MIP_rolling_horizon(self, verbose = False):
        
        start = time.time()

        today = self.Historical - self.T - 1
        self.target = range(today, today + self.T)
        # Days to simulate (including today)
        self.base = today + 0.

        self.res = {}
        for tau in self.target:
            
            # Time window adjustment
            adj_horizon = min(self.horizon_T, self.Historical - today - 1)
            T = range(adj_horizon)
            
            q_s, p_s, d_s, M_kts, K_its, Q_s, F_s = self.gen_sim_paths(today)
            
            for s in range(len(Q_s)):
                Q_s[s] *= 5
                F_s[s] = range(int(len(F_s[s])/4))
            F_s = {s:range(max([len(F_s[ss]) for ss in self.Samples])) for s in self.Samples}
            
            m = gu.Model('TPP+IM_perishable')
            
            ''' Decision Variables '''
            
            ###### TPP variables
            
            # If vehicle v \in F uses arc (i,j) on T, for vsample path s \in Sam
            x = {(i,j,t,v,s):m.addVar(vtype=gu.GRB.BINARY, name=f"x_{i,j,t,v,s}") for s in self.Samples for i in self.V for j in self.V for t in T for v in F_s[s] if i!=j}
            # If supplier i \in M is visited by vehicle v \in F on t \in T, for sample path s \in Sam
            w = {(i,t,v,s):m.addVar(vtype=gu.GRB.BINARY, name=f"w_{i,t,v,s}") for s in self.Samples for i in self.Suppliers for t in T for v in F_s[s]}
            # Quantity of product k \in K bought from supplier i \in M on t \in T, transported in vehicle v \in F, for sample path s \in Sam
            z = {(i,k,t,v,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"z_{i,k,t,v,s}") for s in self.Samples for k in self.Products for t in T for i in M_kts[s][k,t] for v in F_s[s]}
            # Auxiliar variable for routing modeling
            u = {(i,t,v,s):m.addVar(vtype=gu.GRB.BINARY, name=f"u_{i,t,v,s}") for s in self.Samples for i in self.Suppliers for t in T for v in F_s[s]}
            
            ###### Inventory variables
            
            # Quantity of product k \in K aged o \in O_k shipped to final customer on t, for sample path s \in Sam
            y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"y_{k,t,o,s}") for k in self.Products for t in T for o in range(1, self.O_k[k]+1) for s in self.Samples}
            
            ''' State Variables '''
            
            # Quantity of product k \in K that is replenished on t \in T, for sample path s \in Sam
            r = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"r_{k,t,s}") for k in self.Products for t in T for s in self.Samples}
            # Inventory level of product k \in K aged o \in O_k on t \in T, for sample path s \in Sam
            ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"i_{k,t,o,s}") for k in self.Products for t in T for o in range(1, self.O_k[k]+1) for s in self.Samples}
            
            ''' Constraints '''
            for s in self.Samples:
                #Inventory constraints
                ''' For each product k aged o, the inventory level on the first time period is
                the initial inventory level minus what was shipped of it in that same period'''
                for k in self.Products:
                    for o in range(1, self.O_k[k]+1):
                        if o > 1:
                            m.addConstr(ii[k,0,o,s] == self.I_0[k,o]-y[k,0,o,s])
                
                ''' For each product k on t, its inventory level aged 0 is what was replenished on t (r)
                minus what was shipped (y)'''
                for k in self.Products:
                    for t in T:
                        m.addConstr(ii[k,t,1,s] == r[k,t,s]-y[k,t,1,s])
                        
                ''' For each product k, aged o, on t, its inventory level is the inventory level
                aged o-1 on the previous t, minus the amount of it that's shipped '''
                for k in self.Products:
                    for t in T:
                        for o in range(1, self.O_k[k]+1):
                            if t>0 and o>1:
                                m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s]-y[k,t,o,s])                
                
                ''' The amount of product k that is shipped to the final customer on t is equal to its demand'''
                for k in self.Products:
                    for t in T:
                        m.addConstr(gu.quicksum(y[k,t,o,s] for o in range(1, self.O_k[k]+1)) == d_s[s][k,t])
                        
                #TPP constrains
                ''' Modeling of state variable r_kt. For each product k, on each t, its replenishment 
                is the sum of what was bought from all the suppliers'''
                for t in T:
                    for k in self.Products:
                        m.addConstr(gu.quicksum(z[i,k,t,v,s] for i in M_kts[s][k,t] for v in F_s[s]) == r[k,t,s])
                        
                ''' Cannot buy from supplier i more than their available quantity of product k on t '''
                for t in T:
                    for k in self.Products:
                        for i in M_kts[s][k,t]:
                            m.addConstr(gu.quicksum(z[i,k,t,v,s] for v in F_s[s]) <= q_s[s][i,k,t])
                            
                ''' Can only buy product k from supplier i on t and pack it in vehicle v IF
                 vehicle v is visiting the supplier on t'''        
                for t in T:
                    for v in F_s[s]:
                        for k in self.Products:
                            for i in M_kts[s][k,t]:
                                m.addConstr(z[i,k,t,v,s] <= q_s[s][i,k,t]*w[i,t,v,s])
                
                ''' At most, one vehicle visits each supplier each day'''
                for t in T:
                    for i in self.Suppliers:
                        m.addConstr(gu.quicksum(w[i,t,v,s] for v in F_s[s]) <= 1)
                
                ''' Must respect vehicle's capacity '''
                for t in T:
                    for v in F_s[s]:
                        m.addConstr(gu.quicksum(z[i,k,t,v,s] for k in self.Products for i in M_kts[s][k,t] ) <= Q_s[s])
                
                ''' Modeling of variable w_itv: if supplier i is visited by vehicle v on t'''
                for v in F_s[s]:
                    for t in T:
                        for hh in self.Suppliers:
                            m.addConstr(gu.quicksum(x[hh,j,t,v,s] for j in self.V if hh!=j) ==  w[hh,t,v,s])
                            m.addConstr(gu.quicksum(x[i,hh,t,v,s] for i in self.V if i!=hh) ==  w[hh,t,v,s])
                
                ''' Routing modeling - no subtours created'''          
                for t in T:
                    for i in self.Suppliers:
                        for v in F_s[s]:
                            for j in self.Suppliers:
                                if i!=j:
                                    m.addConstr(u[i,t,v,s] - u[j,t,v,s] + len(self.V) * x[i,j,t,v,s] <= len(self.V)-1 )
            
                ''' Non-Anticipativity Constraints'''
                for v in F_s[s]:
                    for i in self.V:
                        for j in self.V:
                            if i!=j:
                                m.addConstr(x[i,j,0,v,s] == gu.quicksum(x[i,j,0,v,s1] for s1 in self.Samples)/self.S)
                
                for v in F_s[s]:
                    for i in self.Suppliers:
                        m.addConstr(w[i,0,v,s] == gu.quicksum(w[i,0,v,s1] for s1 in self.Samples)/self.S)
                
                for v in F_s[s]:
                    for k in self.Products:
                        for i in M_kts[s][k,0]:
                            m.addConstr(z[i,k,0,v,s] == gu.quicksum(z[i,k,0,v,s1] for s1 in self.Samples)/self.S)
                
                for i in self.Suppliers:
                    for v in F_s[s]:
                        m.addConstr(u[i,0,v,s] == gu.quicksum(u[i,0,v,s1] for s1 in self.Samples)/self.S)
                
                for k in self.Products:
                    for o in range(1, self.O_k[k]+1):
                        m.addConstr(y[k,0,o,s] == gu.quicksum(y[k,0,o,s1] for s1 in self.Samples)/self.S)
            
            
            ''' Costs '''
            routes = gu.quicksum(self.c[i,j] * x[i,j,t,v,s] for s in self.Samples for i in self.V for j in self.V for t in T for v in F_s[s] if i!=j)
            acquisition = gu.quicksum(p_s[s][i,k,t]*z[i,k,t,v,s] for s in self.Samples for k in self.Products for t in T for v in F_s[s] for i in M_kts[s][k,t])
            holding = gu.quicksum(self.h[k,t]*ii[k,t,o,s] for s in self.Samples for k in self.Products for t in T for o in range(1, self.O_k[k]+1))
                            
            m.setObjective((routes+acquisition+holding)/self.S)
            
            m.update()
            m.setParam("OutputFlag", verbose)
            m.setParam("MIPGap",0.01)
            m.setParam("TimeLimit", 3600)
            m.optimize()
           
            self.tiempoOpti = time.time() - start
            print(f"Done {tau}")
            
            
            ''' Initial inventory level updated for next iteration '''
            self.I_0 = {(k,o):ii[k,0,o-1,0].X if o > 1 else 0 for k in self.Products for o in range(1, self.O_k[k]+1)}
            
            ''' Saves results for dashboard '''
            self.ii = {(k,o):ii[k,0,o,0].X for k in self.Products for o in range(1, self.O_k[k]+1)}
            self.r = {k:r[k,0,0].X for k in self.Products}
            self.z = {(i,k,v):z[i,k,0,v,0].X for k in self.Products for i in M_kts[0][k,0] for v in F_s[0]}
            self.x = {(i,j,v):x[i,j,0,v,0].X for i in self.V for j in self.V for v in F_s[0] if i!=j}
            self.p_s = {(i,k):sum(p_s[s][i,k,0] for s in self.Samples)/self.S for i in self.Suppliers for k in self.Products}
            self.res[tau] = (ii,r,x,d_s,F_s[0],adj_horizon,p_s,z)
            
            today += 1
        
        print(self.tiempoOpti)
    
    def result_renderization(self, path, save = True, show = True):
        
        images = []
        ''' Rolling Horizon visualization'''

        M_kt, K_it, q, p, d = self.Historic_log

        def unique(lista):
            li = []
            for i in lista:
                if i not in li:
                    li.append(i)
            return li

        for tau in self.target:
            cols = {0:"darkviolet",1:"darkorange",2:"lightcoral",3:"seagreen", 4:"dimgrey", 5:"teal", 6:"olive", 7:"darkmagenta"}
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,ncols=1,figsize=(13,20))
            T = range(self.res[tau][5])
            
            ''' Demand Time Series and Sample Paths '''
            for s in self.Samples:
                ax1.plot([tt for tt in range(tau - self.Historical + self.T + 1, 
                                             tau - self.Historical + self.T + 1 + self.res[tau][5])],
                         [sum(self.res[tau][3][s][k,t] for k in self.Products) for t in T],color=cols[s])
            for tt in range(tau-self.Historical+self.T+2):
                ax1.plot(tt,sum(d[k, self.base+tt] for k in self.Products),"*",color="black",markersize=12)
            ax1.plot([tt for tt in range(tau - self.Historical + self.T + 2)],
                     [sum(d[k, self.base+tt] for k in self.Products) for tt in range(tau-self.Historical+self.T+2)],
                     color="black")
            ax1.set_xlabel("Time period")
            ax1.set_ylabel("Total Demand")
            ax1.set_xticks(ticks=[t for t in range(self.T)])
            ax1.set_xlim(-0.5,self.T-0.5)
            ax1.set_ylim(min([sum(d[k,t] for k in self.Products) for t in range(self.Historical)])-20,max([sum(d[k,t] for k in self.Products) for t in range(self.Historical)])+20)
            
            ''' Total Cost Bars '''
            xx = [tt for tt in range(tau-self.Historical+self.T+2)]
            compra = [sum(self.res[tt+self.Historical-self.T-1][6][i,k]*self.res[tt+self.Historical-self.T-1][7][i,k,v] for i in self.Suppliers for k in self.Products for v in self.res[tt+self.Historical-self.T-1][4] if (i,k,v) in self.res[tt+self.Historical-self.T-1][7]) for tt in xx]
            invent = [sum(self.h[k,tt]*self.res[tt+self.Historical-self.T-1][0][k,o] for k in self.Products for o in range(1, self.O_k[k]+1)) for tt in xx]
            rutas = [sum(self.c[i,j]*self.res[tt+self.Historical-self.T-1][2][i,j,v] for i in self.V for j in self.V for v in self.res[tt+self.Historical-self.T-1][4] if i!=j) for tt in xx]
            ax2.bar(x=xx,height=rutas,bottom=[compra[tt]+invent[tt] for tt in xx],label="Routing",color="sandybrown")
            ax2.bar(x=xx,height=invent,bottom=compra,label="Holding",color="slateblue")
            ax2.bar(x=xx,height=compra,label="Purchase",color="lightseagreen")
            ax2.set_xlabel("Time period")
            ax2.set_ylabel("Total Cost")
            ax2.set_xticks(ticks=[t for t in range(self.T)])
            ax2.set_xlim(-0.5,self.T-0.5)
            a1 = [sum(self.res[tt+self.Historical-self.T-1][6][i,k]*self.res[tt+self.Historical-self.T-1][7][i,k,v] for i in self.Suppliers for k in self.Products for v in self.res[tt+self.Historical-self.T-1][4] if (i,k,v) in self.res[tt+self.Historical-self.T-1][7]) for tt in range(self.T)]
            a2 = [sum(self.h[k,tt]*self.res[tt+self.Historical-self.T-1][0][k,o] for k in self.Products for o in range(1,self.O_k[k]+1)) for tt in range(self.T)]
            a3 = [sum(self.c[i,j]*self.res[tt+self.Historical-self.T-1][2][i,j,v] for i in self.V for j in self.V for v in self.res[tt+self.Historical-self.T-1][4] if i!=j) for tt in range(self.T)]
            lim_cost = max(a1[tt]+a2[tt]+a3[tt] for tt in range(self.T))
            ax2.set_ylim(0,lim_cost+5e3)
            ticks = [i for i in range(0,int(int(lim_cost/1e4+1)*1e4),int(1e4))]
            ax2.set_yticklabels(["${:,.0f}".format(int(i)) for i in ticks])
            ax2.legend(loc="upper right")
            
            ''' Inventory Level by Product '''
            cols_t = {0:(224/255,98/255,88/255),1:(224/255,191/255,76/255),2:(195/255,76/255,224/255),3:(54/255,224/255,129/255),4:(65/255,80/255,224/255),5:(224/255,149/255,103/255),6:(131/255,224/255,114/255),7:(224/255,103/255,91/255),8:(70/255,223/255,224/255),9:(224/255,81/255,183/255)}
            for tt in range(tau-self.Historical+self.T+2):
                list_inv = [round(sum(self.res[tt+self.Historical-self.T-1][0][k,o] for o in range(1,self.O_k[k]+1))) for k in self.Products]
                un_list = unique(list_inv)
                ax3.scatter(x=[tt for i in un_list],y=[i for i in un_list],s=[1e2+(1e4-1e2)*(sum([1 for j in list_inv if j==i])-1)/(self.K-1) for i in un_list],alpha=0.5,color=cols_t[tt])
            ax3.set_xticks(ticks=[t for t in range(self.T)])
            ax3.set_xlabel("Time period")
            ax3.set_ylabel("Inventory Level by product")
            ax3.set_xlim(-0.5,self.T-0.5)
            lim_inv = max([round(sum(self.res[tt+self.Historical-self.T-1][0][k,o] for o in range(1,self.O_k[k]+1))) for k in self.Products for tt in range(int(self.T))])
            ax3.set_ylim(0,lim_inv+1)
            
            ''' Purchased Quantity, by Product '''
            b_purc = []
            for tt in range(tau-self.Historical+self.T+2):
                aver = [round(self.res[tt+self.Historical-self.T-1][1][k]) for k in self.Products]
                b_purc.append(ax4.boxplot(aver,positions=[tt],widths=[0.5],patch_artist=True,flierprops={"marker":'o', "markerfacecolor":cols_t[tt],"markeredgecolor":cols_t[tt]}))
                for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    if item != 'medians':
                        plt.setp(b_purc[-1][item], color=cols_t[tt])
                    else:
                        plt.setp(b_purc[-1][item], color="white")
            ax4.set_xlim(-0.5,self.T-0.5)
            ax4.set_xticks(ticks=[t for t in range(self.T)])
            ax4.set_xlabel("Time period")
            ax4.set_ylabel("Purchased Quantity, by product")
            lim_purc = max([round(self.res[tt+self.Historical-self.T-1][1][k]) for k in self.Products for tt in range(int(self.T))])
            ax4.set_ylim(0,lim_purc+5)
            
            plt.savefig(path+f"RH_d{tau}.png", dpi=300)
            if show:
                plt.show()
            
            if save:
                images.append(imageio.imread(path+f"RH_d{tau}.png"))
        
        if save:
            imageio.mimsave(path+"RH1.gif", images,duration=0.5)
    
        
        
#%% Testing

wd = '/Users/juanbeta/Dropbox/Investigación/IB-MP-IRP-PP/Environment/'

### Uploading deterministic instance
# env = steroid_IRP()
# file = '/Users/juanbeta/Dropbox/Investigación/IB-MP-IRP-PP/Initial Files/Instances4/MVTPP_IM_10_10_5_0.5_1.txt'
# env.upload_params(file)

### Generating sample-paths instance
env = steroid_IRP()
env.MIP_rolling_horizon(verbose = True)
env.result_renderization(wd, show = True, save = True)



        
        