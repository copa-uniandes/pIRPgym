
"""
@author: juanbeta

"""

################################## Modules ##################################
### Basic Librarires
from copy import copy, deepcopy
import pandas as pd
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
        # EDITABLE The return statement can be set to return any of the parameters, historics, 
        # sample paths, etc. of the instance_generator element. 
        self.t = 0

        if self.config['inventory']:
            if self.config['perishability'] == 'ages':
                self.state, self.O_k = self.perish_per_age_inv.reset(inst_gen)

            if return_state:
                return self.state

    # Step 
    def step(self, action:list, inst_gen: instance_generator, validate_action:bool = False, warnings:bool = False):
        if validate_action:
            self.action_validity(action)

        if inst_gen.s_parameters != False:
            '''
            When some parameters are stochastic, the chosen action might not be feasible. Therefore, an aditional intra-step 
            computation must be made and andjustments on the action might be necessary
            '''
            real_action = []
            if self.config['routing']:
                real_action.append(action[0])
                del action[0]

            if self.config['inventory']:
                real_purchase = {(i,k): min(action[0][i,k], inst_gen.W_q[self.t][i,k]) for i in inst_gen.Suppliers for k in inst_gen.Products}
                if self.config['perishability'] == 'ages':
                    real_demand_compliance = self.perish_per_age_inv.get_real_dem_compl(inst_gen, self, real_purchase)

                real_action += [real_purchase, real_demand_compliance]
            
            # EDITABLE Update action due to other stochastic phenomena
            
        else:
            real_action = action

        if inst_gen.other_params['backorders'] == 'backlogs':
            real_action.append(action[-1])

        # Update inventory
        if self.config['inventory']:
            if self.config['perishability'] == 'ages':
                s_tprime, back_orders, perished = self.perish_per_age_inv.update_inventory(inst_gen, self, real_action, warnings)

        # Reward
        reward = []
        if self.config['routing']:
            transport_cost = bl.Routing_management.evaluate_routes(inst_gen, real_action[0])
            reward.append(transport_cost)
        if self.config['inventory']:
            if self.config['perishability'] == 'ages':
                purchase_cost, holding_cost, backorders_cost = self.perish_per_age_inv.compute_costs(inst_gen, self, real_action, s_tprime, perished)
                reward += [purchase_cost, holding_cost, backorders_cost]

        # Time step update and termination check
        self.t += 1
        done = self.check_termination()
        _ = {'backorders': back_orders, 'perished': perished}

        # State update
        if not done:
            if self.config['inventory']:
                self.state = s_tprime
                return self.state, reward, done, real_action, _
            else:
                return None, reward, done, real_action, _
                

    # Checking for episode's termination
    def check_termination(self):
        done = self.t >= self.T
        return done


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