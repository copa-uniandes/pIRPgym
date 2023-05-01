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

    # Initalization method for an instange_generator object
    def __init__(self, look_ahead = ['d','q'], stochastic_params = ['d','q'], historical_data = ['*'],
                  backorders = 'backorders', **kwargs):
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
        self.M = 10                                     # Suppliers
        self.K = 5                                      # Products

        self.T = 7                                      # Horizon

        self.F = 4                                      # Fleet
        self.Q = 40                                     # Vehicles capacity
        self.d_max = 500                                # Max distance per route
                

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
            self.back_l_cost = 600

        ### Extra information ###
        self.other_params = {'look_ahead':look_ahead, 'historical': historical_data, 'backorders': backorders}
        self.s_params = stochastic_params

        ### Custom configurations ###
        utils.assign_env_config(self, kwargs)

        ### Look-ahead parameters
        self.sp_window_sizes = {t:min(self.LA_horizon, self.T - t) for t in range(self.T)}


    # Generates a complete, completely random instance with a given random seed
    def generate_random_instance(self, d_rd_seed:int, s_rd_seed:int, **kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        self.gen_sets()

        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}

        self.O_k = {k:randint(3,self.T+1) for k in self.Products} 
        self.Ages = {k:[i for i in range(1, self.O_k[k] + 1)] for k in self.Products}

        # Offer
        self.M_kt, self.K_it = offer.gen_availabilities(self)
        self.hist_q, self.W_q, self.s_paths_q = offer.gen_quantities(self, **kwargs['q_params'])
        if self.s_paths_q == None: del self.s_paths_q

        self.hist_p, self.W_p, self.s_paths_p = offer.gen_prices(self, **kwargs['p_params'])
        if self.s_paths_p == None: del self.s_paths_p

        # Demand
        self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand(self, **kwargs['d_params'])
        if self.s_paths_d == None: del self.s_paths_d

        # Selling prices
        self.salv_price = selling_prices.gen_salvage_price(self)
        self.opt_price = selling_prices.gen_optimal_price(self)

        self.sell_prices = selling_prices.get_selling_prices(self, kwargs["discount"])
        
        # Inventory
        self.hist_h, self.W_h = costs.gen_h_cost(self, **kwargs['h_params'])

        # Routing
        self.coor, self.c = locations.euclidean_dist_costs(self.V, self.d_rd_seed)

    
    # Generates a basic, completely random instance with a given random seed
    def generate_basic_random_instance(self, d_rd_seed:int = 0, s_rd_seed:int = 1, **kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        self.gen_sets()

        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}

        self.O_k = {k:randint(3,self.T+1) for k in self.Products} 
        self.Ages = {k:[i for i in range(1, self.O_k[k] + 1)] for k in self.Products}

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
        self.coor, self.c = locations.euclidean_dist_costs(self.V, self.d_rd_seed)


    # Generates an CVRPTW instance of the literature
    def CVRP_instance(self, set:str = 'Li', instance:str = 'Li_21.vrp') -> dict[float]:
        self.K:int = 1         # One product
        self.T:int = 1         # One period 
        self.F:int = 100       # 100 vehicles

        self.M, self.Q, self.d_max, self.coor, purchase = locations.upload_cvrp_instance(set, instance)
        purchase = {(i,0):purchase[i] for i in purchase.keys()}

        self.gen_sets()

        self.c = locations.euclidean_distance(self.coor, self.V)

        return purchase        


    # Auxiliary method: Generate iterables of sets
    def gen_sets(self):
        self.Suppliers: range = range(1,self.M + 1);  self.V = range(self.M + 1)
        self.Products: range = range(self.K)
        self.Vehicles: range = range(self.F)
        self.Horizon: range = range(self.T)

        if self.other_params['look_ahead']:
            self.Samples: range = range(self.S)

        if self.other_params['historical']:
            self.TW : range = range(-self.hist_window, self.T)
            self.historical: range = range(-self.hist_window, 0)
        else:
            self.TW: range = self.Horizon
        

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

    ### Holding cost
    def gen_h_cost(inst_gen: instance_generator, **kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 1)
        if kwargs['dist'] == 'd_uniform':   rd_function = randint
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
        W_h = dict()
        for t in inst_gen.Horizon:
            W_h[t] = dict()   
            for k in inst_gen.Products:
                W_h[t][k] = round(rd_function(*kwargs['r_f_params']),2)

                if t < inst_gen.T - 1:
                    hist_h[t+1][k] = hist_h[t][k] + [W_h[t][k]]

        return W_h, hist_h
    


class selling_prices():

    def get_selling_prices(inst_gen:instance_generator, discount) -> dict:
        
        #discount = kwargs["discount"]
        sell_prices = dict()

        if  discount[0] == "no": sell_prices = selling_prices.gen_sell_price_null_discount(inst_gen)
        elif discount[0] == "lin": sell_prices = selling_prices.gen_sell_price_linear_discount(inst_gen)
        elif discount[0] == "mild": sell_prices = selling_prices.gen_sell_price_mild_discount(inst_gen, discount[1])
        elif discount[0] == "strong":sell_prices = selling_prices.gen_sell_price_strong_discount(inst_gen, discount[1])

        return sell_prices
    
    def gen_salvage_price(inst_gen, **kwargs) -> dict:
        salv_price = dict()
        for k in inst_gen.Products:
            li = []
            for t in inst_gen.Horizon:
                for i in inst_gen.Suppliers:
                    if i in inst_gen.M_kt[k,t]:
                        li += [inst_gen.W_p[t][i,k]]
            salv_price[k] = sum(li)/len(li)
        
        return salv_price
    
    def gen_optimal_price(inst_gen, **kwargs) -> dict:
        opt_price = {}
        for k in inst_gen.Products:            
            opt_price[k] = 20*inst_gen.salv_price[k]
        
        return opt_price

    def gen_sell_price_strong_discount(inst_gen: instance_generator, conv_discount) -> dict:

        def ff(k):
            return k*(k+1)/2

        sell_prices = dict()
        for k in inst_gen.Products:
            for o in range(inst_gen.O_k[k] + 1):
                if conv_discount == "conc":
                    if o == inst_gen.O_k[k]: sell_prices[k,o] = inst_gen.salv_price[k]
                    else: sell_prices[k,o] = inst_gen.opt_price[k] - ((inst_gen.opt_price[k]-inst_gen.salv_price[k])*0.25)*(ff(o+1)-1)/(ff(inst_gen.O_k[k])-1)
                elif conv_discount == "conv":
                    if o == 0: sell_prices[k,o] = inst_gen.opt_price[k]
                    else: sell_prices[k,o] = inst_gen.salv_price[k] + ((inst_gen.opt_price[k]-inst_gen.salv_price[k])*0.25)*(ff(inst_gen.O_k[k]-o+1)-1)/(ff(inst_gen.O_k[k])-1)
        
        return sell_prices
        
    def gen_sell_price_mild_discount(inst_gen: instance_generator, conv_discount) -> dict:

        def ff(k):
            return k*(k+1)/2
        
        sell_prices = dict()
        for k in inst_gen.Products:
            for o in range(inst_gen.O_k[k] + 1):
                if conv_discount == "conc":
                    sell_prices[k,o] = inst_gen.salv_price[k] + (inst_gen.opt_price[k]-inst_gen.salv_price[k])*(ff(inst_gen.O_k[k])-ff(o))/ff(inst_gen.O_k[k])
                elif conv_discount == "conv":
                    sell_prices[k,o] = inst_gen.salv_price[k] + (inst_gen.opt_price[k]-inst_gen.salv_price[k])*(ff(inst_gen.O_k[k]-o))/ff(inst_gen.O_k[k])
        
        return sell_prices

    def gen_sell_price_null_discount(inst_gen: instance_generator, **kwargs) -> dict:

        sell_prices = {(k,o):inst_gen.opt_price[k] for k in inst_gen.Products for o in range(inst_gen.O_k[k] + 1)}
        return sell_prices
    
    def gen_sell_price_linear_discount(inst_gen: instance_generator, **kwargs) -> dict:
        
        sell_prices = dict()
        for k in inst_gen.Products:
            for o in range(inst_gen.O_k[k] + 1):
                sell_prices[k,o] = inst_gen.salv_price[k] + (inst_gen.opt_price[k]-inst_gen.salv_price[k])*(inst_gen.O_k[k]-o)/inst_gen.O_k[k]
        
        return sell_prices



class demand():

    ### Demand of products
    def gen_demand(inst_gen: instance_generator, **kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 2)
        if kwargs['dist'] == 'log-normal':   rd_function = lognormal
        elif kwargs['dist'] == 'd_uniform': rd_function = randint


        hist_d = demand.gen_hist_d(inst_gen, rd_function, **kwargs)
        W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)

        if 'd' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            seed(inst_gen.s_rd_seed)
            s_paths_d = demand.gen_empiric_d_sp(inst_gen, hist_d, W_d)
            return hist_d, W_d, s_paths_d

        else:
            return hist_d, W_d, None
    
    # Historic demand
    def gen_hist_d(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]: 
        hist_d = {t:dict() for t in inst_gen.Horizon}
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
        W_d = dict()
        for t in inst_gen.Horizon:
            W_d[t] = dict()   
            for k in inst_gen.Products:
                W_d[t][k] = round(rd_function(*kwargs['r_f_params']),2)

                if t < inst_gen.T - 1:
                    hist_d[t+1][k] = hist_d[t][k] + [W_d[t][k]]

        return W_d, hist_d
    

    # Demand's sample paths
    def gen_empiric_d_sp(inst_gen: instance_generator, hist_d, W_d) -> dict[dict]:
        s_paths_d = dict()
        for t in inst_gen.Horizon: 
            s_paths_d[t] = dict()
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('d' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_d[t][0,sample] = W_d[t]
                else:
                    s_paths_d[t][0,sample] = {k: inst_gen.sim([hist_d[t][k][obs] for obs in range(len(hist_d[t][k])) if hist_d[t][k][obs] > 0]) for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_d[t][day,sample] = {k: inst_gen.sim([hist_d[t][k][obs] for obs in range(len(hist_d[t][k])) if hist_d[t][k][obs] > 0]) for k in inst_gen.Products}

        return s_paths_d
    
    

class offer():
    
    ### Availabilty of products on suppliers
    def gen_availabilities(inst_gen: instance_generator) -> tuple:
        '''
        M_kt: (dict) subset of suppliers that offer k \in K on t \in T
        K_it: (dict) subset of products offered by i \in M on t \in T
        '''
        seed(inst_gen.d_rd_seed + 3)
        M_kt = dict()
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
        seed(inst_gen.d_rd_seed + 4)
        if kwargs['dist'] == 'c_uniform':   rd_function = randint
        hist_q = offer.gen_hist_q(inst_gen, rd_function, **kwargs)
        W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)

        if 'q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            seed(inst_gen.s_rd_seed + 1)
            s_paths_q = offer.gen_empiric_q_sp(inst_gen, hist_q, W_q)
            return hist_q, W_q, s_paths_q 

        else:
            return hist_q, W_q, None


    # Historic availabilities
    def gen_hist_q(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]:
        hist_q = {t:dict() for t in inst_gen.Horizon}
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
        W_q = dict()
        for t in inst_gen.Horizon:
            W_q[t] = dict()  
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
        s_paths_q = dict()
        for t in inst_gen.Horizon: 
            s_paths_q[t] = dict()
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('q' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_q[t][0,sample] = W_q[t]
                else:
                    s_paths_q[t][0,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,0] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]-1):
                    s_paths_q[t][day,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

        return s_paths_q


    ### Prices of products on suppliers
    def gen_prices(inst_gen: instance_generator, **kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 5)
        if kwargs['dist'] == 'd_uniform':   rd_function = randint
        hist_p = offer.gen_hist_p(inst_gen, rd_function, **kwargs)
        W_p, hist_p = offer.gen_W_p(inst_gen, rd_function, hist_p, **kwargs)

        if 'p' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            seed(inst_gen.s_rd_seed + 3)
            s_paths_p = offer.gen_empiric_p_sp(inst_gen, hist_p, W_p)
            return hist_p, W_p, s_paths_p 

        else:
            return hist_p, W_p, None
    
    
    # Historic prices
    def gen_hist_p(inst_gen, rd_function, **kwargs) -> dict[dict]:
        hist_p = {t:dict() for t in inst_gen.Horizon}
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
        W_p = dict()
        for t in inst_gen.Horizon:
            W_p[t] = dict()   
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
        s_paths_p = dict()
        for t in inst_gen.Horizon: 
            s_paths_p[t] = dict()
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('p' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_p[t][0,sample] = W_p[t]
                else:
                    s_paths_p[t][0,sample] = {(i,k): inst_gen.sim([hist_p[t][i,k][obs] for obs in range(len(hist_p[t][i,k])) if hist_p[t][i,k][obs] < 1000]) if i in inst_gen.M_kt[k,0] else 1000 for i in inst_gen.Suppliers for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_p[t][day,sample] = {(i,k): inst_gen.sim([hist_p[t][i,k][obs] for obs in range(len(hist_p[t][i,k])) if hist_p[t][i,k][obs] < 1000]) if i in inst_gen.M_kt[k,t+day] else 1000 for i in inst_gen.Suppliers for k in inst_gen.Products}

        return s_paths_p
    

class locations():

    def generate_grid(V: range): 
        seed()
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid+1), randint(0, size_grid+1)) for i in V}
        return coor, V
    

    def euclidean_distance(coor: dict, V: range):
        # Transportation cost between nodes i and j, estimated using euclidean distance
        return {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in V for j in V if i!=j}
    

    def euclidean_dist_costs(V: range, d_rd_seed):
        seed(d_rd_seed + 6)
        coor, _ = locations.generate_grid(V)
        return coor, locations.euclidean_distance(coor, _)
    
    # Uploading 
    def upload_cvrp_instance(set, instance) -> tuple[int, int, int, dict[float], dict[float]]:
        if set in ['Li','Golden']: CVRPtype = 'dCVRP'; sep = ' '
        elif set == 'Uchoa': CVRPtype = 'CVRP'; sep = '\t'
        file = open(f'./CVRP Instances/{CVRPtype}/{set}/{instance}', mode = 'r');     file = file.readlines()


        line =  int(file[3][13:17]) - 1

        M:int = int(file[3].split(' ')[-1][:-1])        # Number of suppliers
        Q:int = int(file[5].split(' ')[-1][:-1])        # Vehicles capacity

        # Max distance per route
        fila:int = 6
        d_max:int = 1e6   # Max_time
        if file[fila][0]=='D':
            d_max = float(file[fila].split(' ')[-1][:-1])
            fila += 1
        
        # Coordinates
        coor:dict = dict()
        while True:
            fila += 1
            if not file[fila][0] == 'D':
                vals = file[fila].split(sep)
                vals[2] = vals[2][:-1]
                coor[int(vals[0]) - 1] = (float(vals[1]), float(vals[2]))
            else:   break

        # Demand
        purchase:dict = dict()
        while True:
            fila += 1
            if not file[fila][0] == 'D':
                vals = file[fila].split(sep)
                if vals[0] != '1':
                    purchase[float(vals[0]) - 1] = float(vals[1])
            else:   break
        

        return M-1, Q, d_max, coor, purchase
        




    


