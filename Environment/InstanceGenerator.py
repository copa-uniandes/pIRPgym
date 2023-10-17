"""
@author: juanbeta
"""
################################## Modules ##################################
### Basic Librarires
import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
from numpy.random import seed, random, randint, lognormal
import os


class instance_generator():
    ''' Main Instance Generator instance. Generate one of the following instances:
    -  generate_random_instance
    -  generate_basic_random_instance
    -  generate_CundiBoy_instance
    -  upload_CVRP_instance
    '''


    # Initalization method for an instange_generator object
    def __init__(self, look_ahead = ['d'], stochastic_params = False, historical_data = ['*'],
                  backorders = 'backorders', demand_type = "aggregated", **kwargs):
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
        self.other_params = {'look_ahead':look_ahead, 'historical': historical_data, 'backorders': backorders, "demand_type" : demand_type}
        self.s_params = stochastic_params

        ### Custom configurations ###
        assign_env_config(self,kwargs)

        ### Look-ahead parameters
        if look_ahead:
            self.sp_window_sizes = {t:min(self.LA_horizon, self.T - t) for t in range(self.T)}


    # Generates a complete, completely random instance with a given random seed
    def generate_random_instance(self, d_rd_seed:int, s_rd_seed:int, I0:float, **kwargs):
        # Random seeds
        self.d_rd_seed = d_rd_seed
        self.s_rd_seed = s_rd_seed
        
        self.gen_sets()

        # Historical and sample paths arrays
        self.hist_data = {t:{} for t in self.historical}
        self.s_paths = {t:{} for t in self.Horizon}

        #self.O_k = {k:randint(3,self.T+1) for k in self.Products} 
        self.O_k = {k:3 for k in self.Products} 
        self.Ages = {k:[i for i in range(1, self.O_k[k] + 1)] for k in self.Products}

        self.i00 = self.gen_initial_inventory(I0)

        # Offer
        self.M_kt, self.K_it = offer.gen_availabilities(self)
        self.hist_q, self.W_q, self.s_paths_q = offer.gen_quantities(self, **kwargs['q_params'])
        if self.s_paths_q == None: del self.s_paths_q

        self.hist_p, self.W_p, self.s_paths_p = offer.gen_prices(self, **kwargs['p_params'])
        if self.s_paths_p == None: del self.s_paths_p

        # Demand
        if self.other_params["demand_type"] == "aggregated":
            self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand(self, **kwargs['d_params'])
        else:
            self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand_age(self, **kwargs['d_params'])
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
    def generate_basic_random_instance(self, d_rd_seed:int=0, s_rd_seed:int=1, I0:float=0,**kwargs):
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
        if self.other_params["demand_type"] == "aggregated":
            self.hist_d, self.W_d, self.s_paths_d = CundiBoy.demand.gen_demand(self,ex_d,hist_demand,**kwargs['d_params'])
        else:
            self.hist_d, self.W_d, self.s_paths_d = demand.gen_demand_age(self,**kwargs['d_params'])
        if self.s_paths_d == None: del self.s_paths_d

        # Selling prices
        self.salv_price = selling_prices.gen_salvage_price(self)
        self.opt_price = selling_prices.gen_optimal_price(self)

        self.sell_prices = selling_prices.get_selling_prices(self, kwargs["discount"])
        
        # Inventory
        self.hist_h, self.W_h = costs.gen_h_cost(self,**kwargs['h_params'])

        # Routing
        self.coor, self.c = locations.euclidean_dist_costs(self.V,self.d_rd_seed)


    # Generates an CVRPTW instance of the literature
    def upload_CVRP_instance(self, set:str = 'Li', instance:str = 'Li_21.vrp') -> dict[float]:
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
        self.Suppliers:list = list(range(1,self.M + 1));  self.V = [0]+self.Suppliers
        self.Products:list = list(range(self.K))
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

    def get_selling_prices(inst_gen:instance_generator,discount) -> dict:
        
        #discount = kwargs["discount"]
        sell_prices = dict()

        if  discount[0] == "no": sell_prices = selling_prices.gen_sell_price_null_discount(inst_gen)
        elif discount[0] == "lin": sell_prices = selling_prices.gen_sell_price_linear_discount(inst_gen)
        elif discount[0] == "mild": sell_prices = selling_prices.gen_sell_price_mild_discount(inst_gen, discount[1])
        elif discount[0] == "strong": sell_prices = selling_prices.gen_sell_price_strong_discount(inst_gen, discount[1])

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
                    else: 
                        sell_prices[k,o] = inst_gen.opt_price[k] - ((inst_gen.opt_price[k]-inst_gen.salv_price[k])*0.25)*(ff(o+1)-1)/(ff(inst_gen.O_k[k])-1)
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

        if inst_gen.other_params['look_ahead'] != False and ('d' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
            seed(inst_gen.s_rd_seed)
            W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)
            s_paths_d = demand.gen_empiric_d_sp(inst_gen, hist_d, W_d)
            return hist_d, W_d, s_paths_d

        else:
            W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)
            return hist_d, W_d, None
    
    ### Demand of products
    def gen_demand_age(inst_gen: instance_generator, **kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 2)
        if kwargs['dist'] == 'log-normal':   rd_function = lognormal
        elif kwargs['dist'] == 'd_uniform': rd_function = randint

        hist_d = demand.gen_hist_d_age(inst_gen, rd_function, **kwargs)

        if 'd' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            seed(inst_gen.s_rd_seed)
            W_d, hist_d = demand.gen_W_d_age(inst_gen, rd_function, hist_d, **kwargs)
            s_paths_d = demand.gen_empiric_d_sp_age(inst_gen, hist_d, W_d)
            return hist_d, W_d, s_paths_d

        else:
            W_d, hist_d = demand.gen_W_d_age(inst_gen, rd_function, hist_d, **kwargs)
            return hist_d, W_d, None

    # Historic demand
    def gen_hist_d(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]: 
        hist_d = {t:dict() for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('d' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_d[0] = {k:[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products}
        else:
            hist_d[0] = {k:[] for k in inst_gen.Products}

        return hist_d

    # Historic demand for age-dependent demand
    def gen_hist_d_age(inst_gen: instance_generator, rd_function, **kwargs) -> dict[dict]: 
        hist_d = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('d' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            r_f_params = kwargs.get("r_f_params")
            #hist_d[0] = {(k,o):[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}
            hist_d[0] = {(k,o):[round(rd_function(r_f_params[o][0],r_f_params[o][1]),2) for t in inst_gen.historical] for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}
        else:
            hist_d[0] = {(k,o):[] for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}

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
    
    # Realized (real) availabilities
    def gen_W_d_age(inst_gen: instance_generator, rd_function, hist_d, **kwargs) -> tuple:
        '''
        W_d: (dict) demand of k \in K  on t \in T
        '''
        r_f_params = kwargs.get("r_f_params")
        W_d = {}
        for t in inst_gen.Horizon:
            W_d[t] = {}   
            for k in inst_gen.Products:
                for o in range(inst_gen.O_k[k]+1):
                    W_d[t][k,o] = round(rd_function(r_f_params[o][0],r_f_params[o][1]),2)

                    if t < inst_gen.T - 1:
                        hist_d[t+1][k,o] = hist_d[t][k,o] + [W_d[t][k,o]]

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
    
    # Demand's sample paths
    def gen_empiric_d_sp_age(inst_gen: instance_generator, hist_d, W_d) -> dict[dict]:
        s_paths_d = {}
        for t in inst_gen.Horizon: 
            s_paths_d[t] = {}
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('d' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_d[t][0,sample] = W_d[t]
                else:
                    s_paths_d[t][0,sample] = {(k,o): inst_gen.sim([hist_d[t][k,o][obs] for obs in range(len(hist_d[t][k,o])) if hist_d[t][k,o][obs] > 0]) for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_d[t][day,sample] = {(k,o): inst_gen.sim([hist_d[t][k,o][obs] for obs in range(len(hist_d[t][k,o])) if hist_d[t][k,o][obs] > 0]) for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}

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
                sup = randint(3, inst_gen.M+1)
                M_kt[k,t] = list(inst_gen.Suppliers)
                # Random suppliers are removed from subset, regarding {sup}
                for ss in range(inst_gen.M - sup):
                    a = int(randint(0, len(M_kt[k,t])))
                    del M_kt[k,t][a]
        
        # Products offered by each supplier on each time period, based on M_kt
        K_it = {(i,t):[k for k in inst_gen.Products if i in M_kt[k,t]] for i in inst_gen.Suppliers for t in inst_gen.TW}

        return M_kt, K_it
    

    ### Available quantities of products on suppliers
    def gen_quantities(inst_gen:instance_generator,**kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 4)
        if kwargs['dist'] == 'c_uniform':   rd_function = randint
        hist_q = offer.gen_hist_q(inst_gen, rd_function, **kwargs)

        if inst_gen.other_params['look_ahead'] != False and ('q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
            seed(inst_gen.s_rd_seed + 1)
            W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)
            s_paths_q = offer.gen_empiric_q_sp(inst_gen, hist_q, W_q)
            return hist_q, W_q, s_paths_q

        else:
            W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)
            return hist_q, W_q, None


    # Historic availabilities
    def gen_hist_q(inst_gen:instance_generator,rd_function,**kwargs) -> dict[dict]:
        hist_q = {t:dict() for t in inst_gen.Horizon}
        factor = {i:1+random()*2 for i in inst_gen.Suppliers}
        if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_q[0] = {(i,k):[round(rd_function(*kwargs['r_f_params']),2)*factor[i] if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
        else:
            hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

        return hist_q

    
    # Realized (real) availabilities
    def gen_W_q(inst_gen: instance_generator,rd_function,hist_q,**kwargs) -> tuple:
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
    def gen_empiric_q_sp(inst_gen:instance_generator,hist_q,W_q) -> dict[dict]:
        s_paths_q = dict()
        for t in inst_gen.Horizon: 
            s_paths_q[t] = dict()
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
        seed(inst_gen.d_rd_seed + 5)
        if kwargs['dist'] == 'd_uniform':   rd_function = randint
        hist_p = offer.gen_hist_p(inst_gen, rd_function, **kwargs)
        W_p, hist_p = offer.gen_W_p(inst_gen, rd_function, hist_p, **kwargs)

        if inst_gen.other_params['look_ahead'] != False and ('p' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
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
    def gen_empiric_p_sp(inst_gen:instance_generator, hist_p, W_p) -> dict[dict]:
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
        


class CundiBoy():
    def upload_instance():
        data_suppliers = pd.read_excel(os.getcwd()+'/Data/Data_Fruver_0507.xlsx', sheet_name='provider_orders')
        data_demand = pd.read_excel(os.getcwd()+"/Data/Data_Fruver_0507.xlsx",sheet_name="daily_sales_historic")
        data_demand = data_demand[["date","store_product_id","sales"]]

        K = list(pd.unique(data_demand["store_product_id"]))

        data_demand = data_demand.groupby(by=["date","store_product_id"],as_index=False).sum()
        hist_demand = {k:[data_demand.loc[i,"sales"] for i in data_demand.index if data_demand.loc[i,"store_product_id"] == k and data_demand.loc[i,"sales"] > 0] for k in K}
        total_sales = pd.DataFrame.from_dict({k:sum(hist_demand[k]) for k in K},orient="index")
        df_sorted = total_sales.sort_values(by=0, ascending=False)

        # Retrieve the top 30 products with the highest sales
        K = list(df_sorted.head(30).index)
        hist_demand = {k:hist_demand[k] for k in K}


        M = list()
        M_names = list()

        M_k = dict()
        K_i = dict()

        ordered = dict()
        delivered = dict()

        prices = dict()

        for obs in data_suppliers.index:
            i = data_suppliers['provider_id'][obs]
            k = data_suppliers['store_product_id'][obs]

            if k in K:
                if i not in M:
                    M.append(i)
                    M_names.append(data_suppliers["provider_name"])
                    K_i[i] = list()
                
                if k not in M_k.keys():
                    M_k[k] = list()

                M_k[k].append(i)
                K_i[i].append(k)

                if (i,k) not in ordered.keys():
                    ordered[i,k] = 0
                    delivered[i,k] = 0
                    prices[i,k] = list()
                
                ordered[i,k] += data_suppliers['quantity_order'][obs]
                delivered[i,k] += data_suppliers['quantity_received'][obs]

                prices[i,k].append(data_suppliers['cost'][obs])
            

        for i in M:
            K_i[i] = set(K_i[i])
            K_i[i] = list(K_i[i])

        for k in K:
            M_k[k] = set(M_k[k])
            M_k[k] = list(M_k[k])


        service_level = dict()
        for (i,k) in ordered.keys():
            service_level[i,k] = delivered[i,k]/ordered[i,k]

        ex_q = dict()
        ex_d = dict()
        ex_p = dict()
        for k in K:
            ex_d[k] = sum(hist_demand[k])/321

            target_demand = ex_d[k]*1.5
            total_vals = sum([service_level[i,k] for i in M_k[k]])

            for i in M:
                if k in K_i[i]:
                    ex_q[i,k] = target_demand*service_level[i,k]/total_vals
                    ex_p[i,k] = sum(prices[i,k])/len(prices[i,k])
                else:
                    ex_q[i,k] = 0

        return M, M_names, K, M_k, K_i, ex_q, ex_d, ex_p, service_level, hist_demand
    

    class demand():
        ### Demand of products
        def gen_demand(inst_gen:instance_generator,ex_d,hist_demand,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 2)
            if kwargs['dist'] == 'log-normal':   rd_function = lognormal
            elif kwargs['dist'] == 'd_uniform': rd_function = randint

            hist_d = {t:dict() for t in inst_gen.Horizon}
            hist_d.update({0:{k:hist_demand[k] for k in inst_gen.Products}})
            

            if inst_gen.other_params['look_ahead'] != False and ('d' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed)
                W_d, hist_d = CundiBoy.demand.gen_W_d(inst_gen,ex_d,rd_function,hist_d,**kwargs)
                s_paths_d = CundiBoy.demand.gen_empiric_d_sp(inst_gen, hist_d, W_d)
                return hist_d, W_d, s_paths_d

            else:
                W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)
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
        def gen_W_d(inst_gen:instance_generator,ex_d,rd_function,hist_d,**kwargs) -> tuple:
            '''
            W_d: (dict) demand of k \in K  on t \in T
            '''
            W_d = dict()
            for t in inst_gen.Horizon:
                W_d[t] = dict()
                for k in inst_gen.Products:
                    mean_parameter = np.log(ex_d[k]) - 0.5 * np.log(1 + kwargs['r_f_params'] / ex_d[k]**2)
                    sigma = np.sqrt(np.log(1 + kwargs['r_f_params'] / ex_d[k]**2))
                    W_d[t][k] = lognormal(mean_parameter,sigma)

                    if t < inst_gen.T - 1:
                        hist_d[t+1][k] = hist_d[t][k] +[W_d[t][k]]

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
        ### Available quantities of products on suppliers
        def gen_quantities(inst_gen:instance_generator,ex_q,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 4)
            if kwargs['dist'] == 'c_uniform':   rd_function = randint
            hist_q = CundiBoy.offer.gen_hist_q(inst_gen,ex_q,rd_function,**kwargs)

            if inst_gen.other_params['look_ahead'] != False and ('q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed + 1)
                W_q, hist_q = CundiBoy.offer.gen_W_q(inst_gen, ex_q, rd_function, hist_q, **kwargs)
                s_paths_q = CundiBoy.offer.gen_empiric_q_sp(inst_gen, hist_q, W_q)
                return hist_q, W_q, s_paths_q 

            else:
                W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)
                return hist_q, W_q, None


        # Historic availabilities
        def gen_hist_q(inst_gen:instance_generator,ex_q,rd_function,**kwargs) -> dict[dict]:
            hist_q = {t:dict() for t in inst_gen.Horizon}
            if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
                hist_q[0] = {(i,k):[max(round(rd_function(ex_q[i,k]-kwargs['r_f_params'],ex_q[i,k]+kwargs['r_f_params']),2),0) if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
            else:
                hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

            return hist_q

        
        # Realized (real) availabilities
        def gen_W_q(inst_gen:instance_generator,ex_q,rd_function,hist_q,**kwargs) -> tuple:
            '''
            W_q: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            '''
            W_q = dict()
            for t in inst_gen.Horizon:
                W_q[t] = dict()  
                for i in inst_gen.Suppliers:
                    for k in inst_gen.Products:
                        if i in inst_gen.M_kt[k,t]:
                            W_q[t][(i,k)] = max(round(rd_function(ex_q[i,k]-kwargs['r_f_params'],ex_q[i,k]+kwargs['r_f_params']),2),0)
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

                    for day in range(1,inst_gen.sp_window_sizes[t]):
                        s_paths_q[t][day,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

            return s_paths_q


        ### Prices of products on suppliers
        def gen_prices(inst_gen:instance_generator,ex_p,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 5)
            if kwargs['dist'] == 'd_uniform':   rd_function = randint
            hist_p = CundiBoy.offer.gen_hist_p(inst_gen,rd_function,ex_p,**kwargs)
            W_p, hist_p = CundiBoy.offer.gen_W_p(inst_gen,rd_function, hist_p,ex_p,**kwargs)

            if inst_gen.other_params['look_ahead'] != False and ('p' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed + 3)
                s_paths_p = CundiBoy.offer.gen_empiric_p_sp(inst_gen, hist_p, W_p)
                return hist_p, W_p, s_paths_p 

            else:
                return hist_p, W_p, None
        
        
        # Historic prices
        def gen_hist_p(inst_gen,rd_function,ex_p,**kwargs) -> dict[dict]:
            hist_p = {t:dict() for t in inst_gen.Horizon}
            if inst_gen.other_params['historical'] != False and ('p' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
                hist_p[0] = {(i,k):[round(rd_function(ex_p[i,k]*(1-kwargs['r_f_params']),ex_p[i,k]*(1+kwargs['r_f_params'])),2) if i in inst_gen.M_kt[k,t] else 1000 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
            else:
                hist_p[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

            return hist_p


        # Realized (real) prices
        def gen_W_p(inst_gen:instance_generator,rd_function,hist_p,ex_p,**kwargs) -> tuple:
            '''
            W_p: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            '''
            W_p = dict()
            for t in inst_gen.Horizon:
                W_p[t] = dict()   
                for i in inst_gen.Suppliers:
                    for k in inst_gen.Products:
                        if i in inst_gen.M_kt[k,t]:
                            W_p[t][(i,k)] = round(rd_function(ex_p[i,k]*(1-kwargs['r_f_params']),ex_p[i,k]*(1+kwargs['r_f_params'])),2)
                        else:   W_p[t][(i,k)] = 10000

                        if t < inst_gen.T - 1:
                            hist_p[t+1][i,k] = hist_p[t][i,k] + [W_p[t][i,k]]

            return W_p, hist_p





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