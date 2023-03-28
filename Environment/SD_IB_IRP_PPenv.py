
"""
@author: juanbeta

! stochastic parameters

WARNINGS:
! Q parameter: Not well defined
! Check HOLDING COST (TIMING)

"""

################################## Modules ##################################
### Basic Librarires
from copy import copy, deepcopy
import pandas as pd
from termcolor import colored
import gym

### Instance generator
from InstanceGenerator import instance_generator


### Building blocks
import BuildingBlocks as bl

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

class steroid_IRP(gym.Env, bl.Inventory_management): 
    
    # Initialization method
    def __init__(self, routing = True, inventory = True, perishability = True):
        
        assert inventory >= bool(perishability), 'Perishability only available with Inventory Problem'
        self.config = {'routing': routing, 'inventory': inventory, 'perishability': perishability}
        


    # Reseting the environment
    def reset(self, inst_gen: instance_generator, return_state:bool = False):
        '''
        Reseting the environment. Genrate or upload the instance.
        PARAMETER:
        return_state: Indicates whether the state is returned
        '''  
        self.t = 0

        if self.config['inventory']:
            if self.config['perishability'] == 'ages':
                self.state, self.O_k = self.perishable_per_age_inv.reset(inst_gen)

        


        



        # ########################### Provisional transition structure ##########################
        # # Deterministic parameters
        # self.gen_sets(inst_gen)
        # self.other_env_params = inst_gen.other_params
        # self.stochastic_parameters = inst_gen.s_params

        # self.O_k = inst_gen.gen_ages()
        
        # self.c = inst_gen.c

        # # Availabilities
        # self.M_kt, self.K_it = inst_gen.M_kt, inst_gen.K_it

        # # Other deterministic parameters
        # self.p_t = inst_gen.W_p
        # self.h_t = inst_gen.W_h


        # # Recovery of other information
        # self.exog_info = {t:{'q': inst_gen.W_q[t], 'd': inst_gen.W_d[t]} for t in self.Horizon}

        # if self.other_env_params['backorders'] == 'backorders':
        #     self.back_o_cost = inst_gen.back_o_cost

        # if self.other_env_params['look_ahead']:
        #     self.S = inst_gen.S              # Number of sample paths
        #     self.LA_horizon = inst_gen.LA_horizon     # Look-ahead time window's size (includes current period)
        #     self.window_sizes = inst_gen.sp_window_sizes
        #     self.hor_sample_paths = {t:{'q': inst_gen.s_paths_q[t], 'd': inst_gen.s_paths_d[t]} for t in self.Horizon}
        
        # if self.other_env_params['historical']:
        #     self.hor_historical_data = {t:{'q': inst_gen.hist_q[t], 'd': inst_gen.hist_d[t]} for t in self.Horizon} 

        # ## State ##
        # self.state = {(k,o):0   for k in self.Products for o in range(1, self.O_k[k] + 1)}
        # if self.other_env_params['backorders'] == 'backlogs':
        #     for k in self.Products:
        #         self.state[k,'B'] = 0
        
        # # Current values
        # self.p = self.p_t[self.t]
        # self.h = self.h_t[self.t]

        # self.W_t = self.exog_info[self.t]

        # self.d = self.W_t['d'] 
        # self.q = self.W_t['q'] 

        # self.sample_paths = self.hor_sample_paths[self.t] 
        # self.historical_data = self.hor_historical_data[self.t] 

        if return_state:
            _ = {'sample_paths': self.sample_paths, 
                 'sample_path_window_size': self.window_sizes[self.t]}
            return self.state, _
                              




    # Step 
    def step(self, action:list, validate_action:bool = False, warnings:bool = False):
        if validate_action:
            self.action_validity(action)

        if self.stochastic_parameters != False:
            self.q = self.W_t['q']; self.d = self.W_t['d']
            real_action = self.get_real_action(action, fixed_compl=False)
        else:
            real_action = action

        # Inventory dynamics
        s_tprime, back_orders, perished = self.transition_function(real_action, self.W_t, warnings)

        # Reward
        transport_cost, purchase_cost, holding_cost, backorders_cost = self.compute_costs(real_action, s_tprime, perished)
        reward = [transport_cost, purchase_cost, holding_cost, backorders_cost]

        # Time step update and termination check
        self.t += 1
        done = self.check_termination()
        _ = {'backorders': back_orders, 'perished': perished}

        # State update
        if not done:
            self.update_state(s_tprime)
    
            # EXTRA INFORMATION TO BE RETURNED
            _ = {'p': self.p, 'q': self.q, 'h': self.h, 'd': self.d, 'backorders': back_orders, 
                 'sample_path_window_size': self.window_sizes[self.t], 'perished': perished}
            if self.other_env_params['historical']:
                _['historical_info'] = self.historical_data
            if self.other_env_params['look_ahead']:
                _['sample_paths'] = self.sample_paths
        else:
            self.state = s_tprime

        return self.state, reward, done, real_action, _


    def get_real_action(self, action, fixed_compl=True):
        '''
        When some parameters are stochastic, the chosen action might not be feasible.
        Therefore, an aditional intra-step computation must be made and andjustments 
        on the action might be necessary

        '''
        purchase, demand_compliance = action [1:3]

        # The purchase exceeds the available quantities of the suppliers
        real_purchase = {(i,k): min(purchase[i,k], self.q[i,k]) for i in self.Suppliers for k in self.Products}

        if fixed_compl:
            real_demand_compliance = copy(demand_compliance)
            for k in self.Products:
                # The demand compliance of purchased items differs from the purchase 
                real_demand_compliance[k,0] = min(real_demand_compliance[k,0], sum(real_purchase[i,k] for i in self.Suppliers))
                # The demand is lower than the demand compliance plan 
                if sum(real_demand_compliance[k,o] for o in range(self.O_k[k] + 1)) > self.d[k]:
                    demand = self.d[k]
                    age = self.O_k[k]
                    while demand > 0:
                        if demand >= real_demand_compliance[k,age]:
                            demand -= real_demand_compliance[k,age]
                            age -= 1
                        elif demand < real_demand_compliance[k,age]:
                            real_demand_compliance[k,age] = demand
                            break
        else:
            real_demand_compliance={}
            for k in self.Products:
                left_to_comply = self.d[k]
                for o in range(self.O_k[k],0,-1):
                    if self.stochastic_parameters:
                        real_demand_compliance[k,o] = min(self.state[k,o], left_to_comply)
                        left_to_comply -= real_demand_compliance[k,o]
                    else:
                        real_demand_compliance[k,o] = 0
                
                real_demand_compliance[k,0] = min(sum(real_purchase[i,k] for i in self.Suppliers), left_to_comply)

        real_action = [action[0], real_purchase, real_demand_compliance]

        return real_action


    # Compute costs of a given procurement plan for a given day
    def compute_costs(self, action, s_tprime, perished):
        routes, purchase, demand_compliance = action[:3]
        if self.other_env_params['backorders'] == 'backlogs':   back_o_compliance = action[3]

        transport_cost = 0
        for route in routes:
            transport_cost += sum(self.c[route[i], route[i + 1]] for i in range(len(route) - 1))
        
        purchase_cost = sum(purchase[i,k] * self.p[i,k]   for i in self.Suppliers for k in self.Products)
        
        # TODO!!!!!
        holding_cost = sum(sum(s_tprime[k,o] for o in range(1, self.O_k[k] + 1)) * self.h[k] for k in self.Products)
        for k in perished.keys():
            holding_cost += perished[k] * self.h[k]

        backorders_cost = 0
        if self.other_env_params['backorders'] == 'backorders':
            backorders = sum(max(self.d[k] - sum(demand_compliance[k,o] for o in range(self.O_k[k]+1)),0) for k in self.Products)
            backorders_cost = backorders * self.back_o_cost
        
        elif self.other_env_params['backorders'] == 'backlogs':
            backorders_cost = sum(s_tprime[k,'B'] for k in self.Products) * self.back_l_cost

        return transport_cost, purchase_cost, holding_cost, backorders_cost
            
    
    # Inventory dynamics of the environment
    def transition_function(self, real_action, W, warnings):
        purchase, demand_compliance = real_action[1:3]
        # backlogs
        if self.other_env_params['backorders'] == 'backlogs':
            back_o_compliance = real_action[3]
        inventory = deepcopy(self.state)
        back_orders = {}
        perished = {}

        # Inventory update
        for k in self.Products:
            inventory[k,1] = round(sum(purchase[i,k] for i in self.Suppliers) - demand_compliance[k,0],2)

            max_age = self.O_k[k]
            if max_age > 1:
                for o in range(2, max_age + 1):
                        inventory[k,o] = round(self.state[k,o - 1] - demand_compliance[k,o - 1],2)
            
            if self.other_env_params['backorders'] == 'backorders' and sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) < W['d'][k]:
                back_orders[k] = round(W['d'][k] - sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)),2)

            if self.other_env_params['backorders'] == 'backlogs':
                new_backlogs = round(max(self.W['d'][k] - sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)),0),2)
                inventory[k,'B'] = round(self.state[k,'B'] + new_backlogs - sum(back_o_compliance[k,o] for o in range(self.O_k[k]+1)),2)
            
            if self.state[k, max_age] - demand_compliance[k,max_age] > 0:
                    perished[k] = self.state[k, max_age] - demand_compliance[k,max_age]
    

            # Factibility checks         
            if warnings:
                if self.state[k, max_age] - demand_compliance[k,max_age] > 0:
                    # reward += self.penalization_cost
                    print(colored(f'Warning! {self.state[k, max_age] - demand_compliance[k,max_age]} units of {k} were lost due to perishability','yellow'))
    

                if sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) < W['d'][k]:
                    print(colored(f'Warning! Demand of product {k} was not fulfilled', 'yellow'))

        return inventory, back_orders, perished


    # Checking for episode's termination
    def check_termination(self):
        done = self.t >= self.T
        return done


    def update_state(self, s_tprime):
        # Update deterministic parameters
        self.p = self.p_t[self.t]
        self.h = self.h_t[self.t]

        # Update historicalals
        self.historical_data = self.hor_historical_data[self.t]

        # Update sample pahts
        self.sample_paths = self.hor_sample_paths[self.t]

        # Update exogenous information
        self.W_t = self.exog_info[self.t]

        self.state = s_tprime


    def action_validity(self, action):
        routes, purchase, demand_compliance = action[:3]
        if self.other_env_params['backorders'] == 'backlogs':   back_o_compliance = action[3]
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


    # Simple function to visualize the inventory
    def print_inventory(self):
        max_O = max([self.O_k[k] for k in self.Products])
        listamax = [[self.state[k,o] for o in self.Ages[k]] for k in self.Products]
        df = pd.DataFrame(listamax, index=pd.Index([str(k) for k in self.Products], name='Products'),
        columns=pd.Index([str(o) for o in range(1, max_O + 1)], name='Ages'))

        return df


    def print_state(self):
        print(f'################################### STEP {self.t} ###################################')
        print('INVENTORY')
        max_age = max(list(self.O_k.values()))
        string = 'M \ O \t '
        for o in range(1, max_age + 1):
            string += f'{o} \t'
        print(string)
        for k in self.Products:
            string = f'S{k} \t '
            for o in self.Ages[k]:
                string += f'{self.state[k,o]} \t'
            print(string)
        
        print('\n')
        print('DEMAND')
        string1 = 'K'
        string2 = 'd'
        for k in self.Products:
            string1 += f'\t{k}'
            string2 += f'\t{self.W_t["d"][k]}'
        print(string1)
        print(string2)

        print('\n')
        print('AVAILABLE QUANTITIES')
        string = 'M\K \t'
        for k in self.Products:
            string += f'{k} \t'
        print(string)
        for i in self.Suppliers:
            new_string = f'{i}\t'
            for k in self.Products:
                if self.W_t['q'][i,k] == 0:
                    new_string += f'{self.W_t["q"][i,k]}\t'
                else:
                    new_string += f'{self.W_t["q"][i,k]}\t'
            print(new_string)


    # Printing a representation of the environment (repr(env))
    def __repr__(self):
        return f'Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance. V = {self.M}; K = {self.K}; F = {self.F}'


    ##################### EXTRA Non-functional features #####################
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