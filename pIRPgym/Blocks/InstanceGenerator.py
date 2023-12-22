"""
@author: juanbeta
"""
################################## Modules ##################################
### Basic Librarires
import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
from numpy.random import randint
import os
from typing import Union

from .InstanceGeneration.demand import demand
from .InstanceGeneration.costs import costs
from .InstanceGeneration.selling_prices import selling_prices
from .InstanceGeneration.offer import offer
from .InstanceGeneration.locations import locations
from .InstanceGeneration.CundiBoy import CundiBoy
from .InstanceGeneration.environmental_indicators import indicators

class instance_generator():
    ''' Main Instance Generator instance. Generate one of the following instances:
    -  generate_random_instance
    -  generate_basic_random_instance
    -  generate_CundiBoy_instance
    -  upload_CVRP_instance
    '''
    options = ['random','basic_random','CundiBoy','CVRP']


    # Initalization method for an instange_generator object
    def __init__(self,look_ahead=['d'],stochastic_params:Union[list[str],bool]=False,
                 historical_data:Union[list[str],bool]=['*'],backorders = 'backorders', 
                 demand_type = "aggregated",sustainability=False, **kwargs):
        '''
        Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance
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
        3.  False: No historical data will be generated
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
            
            
            S = 4:  Number of sample paths 
            LA_horizon = 5: Number of look-ahead periods
        '''
        
        ### Main parameters ###
        self.M = 15                                     # Suppliers
        self.K = 5                                      # Products

        self.T = 7                                      # Horizon

        self.F = 15                                     # Fleet
        self.Q = 40                                     # Vehicles capacity
        self.d_max = 500                                # Max distance per route

        self.theta = 0.8                                # Service Level requirement
        self.hold_cost = True                           # Whether to include the holding cost in the objectives or not
        self.rr = 0.1                                   # Minimum order quantity per supplier
        self.gamma = 1                                  # Time discounting factor
        self.E = ["climate","water","land","fossil"]    # Environmental Indicators
        self.metric_names = {"costs":"Total\nCost","transport cost":"Transportation\nCost", "purchase cost":"Purchasing\nCost", "holding cost":"Holding\nCost", "backorders cost":"Backorders\nCost",
                             "climate":"Climate\nChange", "water":"Water\nUse", "land":"Land\nUse", "fossil":"Fossil Fuel\nDepletion"}
        self.metric_units = {"costs":"$","transport cost":"$", "purchase cost":"$", "holding cost":"$", "backorders cost":"$",
                             "climate":"kg CO2 eq", "water":"m3 depriv.", "land":"Pt.", "fossil":"MJ"}

        ### Look-ahead parameters ###
        if look_ahead:    
            self.S = 4              # Number of sample paths
            self.LA_horizon = 5     # Look-ahead time window's size (includes current period)
        
        ### historical log parameters ###
        if historical_data:        
            self.hist_window = 40       # historical window

        ### Backorders parameters ###
        if backorders == 'backlogs':
            self.back_l_cost = 600

        ### Extra information ###
        self.other_params = {'look_ahead':look_ahead, 'historical': historical_data, 'backorders': backorders, "demand_type" : demand_type}
        self.s_params = stochastic_params
        self.sustainability = sustainability

        ### Custom configurations ###
        assign_env_config(self,kwargs)

        ### Look-ahead parameters
        if look_ahead:
            self.sp_window_sizes = {t:min(self.LA_horizon, self.T - t) for t in range(self.T)}


    # Generates a complete, completely random instance with a given random seed
    def generate_random_instance(self,d_rd_seed:int,s_rd_seed:int,I0:float=0, **kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        self.gen_sets()

        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}

        ages = [3,6,5,3,6,4,4]
        if len(self.Products) <= 7: self.O_k = dict(zip(self.Products,ages[:len(self.Products)]))
        else: self.O_k = {k:3 for k in self.Products}
        self.Ages = {k:[i for i in range(1, self.O_k[k] + 1)] for k in self.Products}

        self.i00 = self.gen_initial_inventory(I0)

        # Routing
        self.coor, self.c = locations.euclidean_dist_costs(self.V, self.d_rd_seed, self.sustainability)

        # Supply
        self.M_kt, self.K_it = offer.gen_availabilities(self)
        self.hist_q, self.W_q, self.s_paths_q = offer.gen_quantities(self,**kwargs['q_params'])
        if self.s_paths_q == None: del self.s_paths_q

        # Purchase
        self.hist_p, self.W_p, self.s_paths_p = offer.gen_prices(self,**kwargs['p_params'])
        if self.s_paths_p == None: del self.s_paths_p

        # Inventory
        self.hist_h, self.W_h = costs.gen_h_cost(self, **kwargs['h_params'])

        # Backorders
        self.prof_margin = costs.gen_profit_margin(self); self.back_o_cost = costs.gen_backo_cost(self)

        # Demand
        self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand(self,**kwargs['d_params'])
        if self.s_paths_d == None: del self.s_paths_d

        # Environmental Indicators
        if self.sustainability: self.c_LCA, self.h_LCA, self.waste_LCA = indicators.get_environmental_indicators(self)

    
    # Generates a basic, completely random instance with a given random seed
    def generate_basic_random_instance(self,d_rd_seed:int=0,s_rd_seed:int=1,I0:float=0,**kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        self.gen_sets()

        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}

        self.O_k = {k:randint(3,self.T+1) for k in self.Products}
        self.Ages = {k:[i for i in range(1, self.O_k[k] + 1)] for k in self.Products}

        self.i00 = self.gen_initial_inventory(I0)

        # Supply
        self.M_kt, self.K_it = offer.gen_availabilities(self)
        self.hist_q, self.W_q, self.s_paths_q = offer.gen_quantities(self, **kwargs['q_params'])
        if self.s_paths_q == None: del self.s_paths_q

        self.hist_p, self.W_p, self.s_paths_p = offer.gen_prices(self, **kwargs['p_params'])
        if self.s_paths_p == None: del self.s_paths_p

        # Demand
        self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand(self, **kwargs['d_params'])
        if self.s_paths_d == None: del self.s_paths_d

        # Backorders
        self.prof_margin = costs.gen_profit_margin(self)
        self.back_o_cost = costs.gen_backo_cost(self)
        
        # Inventory
        self.hist_h, self.W_h = costs.gen_h_cost(self, **kwargs['h_params'])

        # Routing
        self.coor,self.c = locations.euclidean_dist_costs(self.V, self.d_rd_seed)
        self.d_max = max(self.d_max,max([2*value for value in self.c.values()]))
        if self.sustainability: self.c_LCA, self.h_LCA, self.waste_LCA = indicators.get_environmental_indicators(self)


    # Generates a (dummy) instance of CundiBoy
    def generate_CundiBoy_instance(self,d_rd_seed:int=0,s_rd_seed:int=1,I0:float=0,**kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        ''' Provitional '''
        M, M_names, K, M_k, K_i, ex_q, ex_d, ex_p, service_level, hist_demand = CundiBoy.upload_instance()

        self.gen_sets()
        self.Suppliers = M[:self.M]
        self.Products = K[:self.K]
        self.V = [0] + self.Suppliers


        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}

        self.O_k = {k:randint(3,self.T+1) for k in self.Products} 
        self.Ages = {k:[i for i in range(1, self.O_k[k] + 1)] for k in self.Products}

        self.i00 = self.gen_initial_inventory(I0)

        # Offer
        self.M_kt = {(k,t):[i for i in M_k[k] if i in self.Suppliers] for t in self.TW for k in self.Products}; self.K_it = {(i,t):[k for k in K_i[i] if k in self.Products] for t in self.TW for i in self.Suppliers}
        self.hist_q, self.W_q, self.s_paths_q = CundiBoy.offer.gen_quantities(self,ex_q,**kwargs['q_params'])
        self.ex_q = ex_q
        if self.s_paths_q == None: del self.s_paths_q
        
        self.hist_p, self.W_p, self.s_paths_p = CundiBoy.offer.gen_prices(self,ex_p,**kwargs['p_params'])
        if self.s_paths_p == None: del self.s_paths_p

        # Demand
        self.hist_d, self.W_d, self.s_paths_d = CundiBoy.demand.gen_demand(self,ex_d,hist_demand,**kwargs['d_params'])
        if self.s_paths_d == None: del self.s_paths_d
        
        # Inventory
        self.hist_h, self.W_h = costs.gen_h_cost(self,**kwargs['h_params'])

        # Routing
        self.coor, self.c = locations.euclidean_dist_costs(self.V,self.d_rd_seed)
        if self.sustainability: self.c_LCA, self.h_LCA, self.waste_LCA = indicators.get_environmental_indicators(self)


    # Generates an CVRPTW instance of the literature
    def upload_CVRP_instance(self, set:str = 'Li', instance:str = 'Li_21.vrp') -> tuple:
        self.K:int = 1         # One product
        self.T:int = 1         # One period 
        

        self.M,self.Q,self.d_max,self.coor,purchase,benchmark = locations.upload_cvrp_instance(set, instance)
        self.F:int = self.M       # M vehicles
        # purchase = {(i):purchase[i] for i in purchase.keys()}

        self.gen_sets()

        self.c = locations.euclidean_distance(self.coor,self.V)


        return purchase,benchmark  


    # Auxiliary method: Generate iterables of sets
    def gen_sets(self):
        self.Suppliers:list = list(range(1,self.M + 1));  self.V = [0]+self.Suppliers; self.A = [(i,j) for i in self.V for j in self.V if i != j]
        self.Products:list = list(range(1,self.K+1))
        self.Vehicles: range = range(self.F)
        self.Horizon: range = range(self.T)

        if self.other_params['look_ahead']:
            self.Samples: range = range(self.S)

        if self.other_params['historical']:
            self.TW : range = range(-self.hist_window, self.T)
            self.historical: range = range(-self.hist_window, 0)
        else:
            self.TW: range = self.Horizon
    

    def gen_initial_inventory(self,a):

        i00 = {(k,o):a for k in self.Products for o in self.Ages[k]}

        return i00


    ''' Auxiliary method to assign custom configurations the instance '''
def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")