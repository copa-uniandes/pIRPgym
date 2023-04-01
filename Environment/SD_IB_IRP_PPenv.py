
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
from BuildingBlocks import Routing_management, Inventory_management 

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
                self.state, self.O_k, self.Ages = Inventory_management.perish_per_age_inv.reset(inst_gen)
                

            if return_state:
                return self.state


    # Step 
    def step(self, action:list, inst_gen: instance_generator, validate_action:bool = False, warnings:bool = False):
        if validate_action:
            self.action_validity(action, inst_gen)

        if inst_gen.s_params != False:
            '''
            When some parameters are stochastic, the chosen action might not be feasible. Therefore, an aditional intra-step 
            computation must be made and andjustments on the action might be necessary
            '''
            real_action = []
            if self.config['routing']:
                real_routing = action[0]
                real_purchase = {(i,k): min(action[1][i,k], inst_gen.W_q[self.t][i,k]) for i in inst_gen.Suppliers for k in inst_gen.Products}

            if self.config['inventory']:
                if not self.config['routing']:
                    real_purchase = {(i,k): min(action[0][i,k], inst_gen.W_q[self.t][i,k]) for i in inst_gen.Suppliers for k in inst_gen.Products}
                if self.config['perishability'] == 'ages':
                    real_demand_compliance = Inventory_management.perish_per_age_inv.get_real_dem_compl_FIFO(inst_gen, self, real_purchase)
            
        else:
            if self.config['routing']:
                real_routing, real_purchase  = action[0:2]
            
            if self.config['inventory']:
                if not self.config['routing']:
                        real_purchase, real_demand_compliance  = action[0:2]
                else:
                    real_demand_compliance = action[2]

        if inst_gen.other_params['backorders'] == 'backlogs':
            real_back_o_compliance = action[-1]

        # Update inventory
        if self.config['inventory']:
            if self.config['perishability'] == 'ages':
                s_tprime, back_orders, perished = Inventory_management.perish_per_age_inv.update_inventory(inst_gen, self, real_purchase, real_demand_compliance, warnings)

        # Reward
        reward = []
        if self.config['routing']:
            transport_cost = Routing_management.evaluate_routes(inst_gen, real_routing)
            reward.append(transport_cost)
        if self.config['inventory']:
            if self.config['perishability'] == 'ages':
                purchase_cost, holding_cost, backorders_cost = Inventory_management.perish_per_age_inv.compute_costs(inst_gen, self, real_purchase, real_demand_compliance, s_tprime, perished)
                reward += [purchase_cost, holding_cost, backorders_cost]

        # Time step update and termination check
        self.t += 1
        done = self.check_termination(inst_gen)
        if self.config['inventory']:
            _ = {'backorders': back_orders, 'perished': perished}
        else:
            _ = {}

        # Action assembly
        real_action = []
        if self.config['routing']:
            real_action += [real_routing, real_purchase]
            if self.config['inventory']:
                real_action.append(real_demand_compliance)
        elif self.config['inventory']:
            real_action += [real_purchase, real_demand_compliance]

        # State update
        if not done:
            if self.config['inventory']:
                self.state = s_tprime
                return self.state, reward, done, real_action, _
            else:
                return None, reward, done, real_action, _
        else:
            return None, reward, done, real_action, _

    # Checking for episode's termination
    def check_termination(self, inst_gen: instance_generator):
        done = self.t >= inst_gen.T
        return done


    # Method to evaluate actions
    def action_validity(self, action, inst_gen: instance_generator):
        # TODO: Manage action by individual componentes, problem when deleting components 
        if self.config['routing']:
            routes = action[0]
            # Routing check
            assert not len(routes) > inst_gen.F, 'The number of routes exceedes the number of vehicles'

            for route in routes:
                assert not (route[0] != 0 or route[-1] != 0), \
                    'Routes not valid, must start and end at the depot'

                route_capacity = sum(action[1][node,k] for k in inst_gen.Products for node in route[1:-2])
                assert not route_capacity > inst_gen.Q, \
                    "Purchased items exceed vehicle's capacity"

                assert not len(set(route)) != len(route) - 1, \
                    'Suppliers can only be visited once by a route'

                for i in range(len(route)):
                    assert not route[i] not in inst_gen.V, \
                        'Route must be composed of existing suppliers'
            
            #del action[0]   
         
        if self.config['inventory']: 
            purchase, demand_compliance = action[0:2]
            if inst_gen.other_params['backorders'] == 'backlogs':   back_o_compliance = action[3]

            # Purchase
            for i in inst_gen.Suppliers:
                for k in inst_gen.Products:
                    assert not purchase[i,k] > inst_gen.W_q[self.t][i,k], \
                        f"Purchased quantities exceed suppliers' available quantities  ({i},{k})"
            
            # Demand_compliance
            for k in inst_gen.Products:
                assert not (inst_gen.other_params['backorders'] != 'backlogs' and demand_compliance[k,0] > sum(purchase[i,k] for i in inst_gen.Suppliers)), \
                    f'Demand compliance with purchased items of product {k} exceed the purchase'

                assert not (self.others['backorders'] == 'backlogs' and demand_compliance[k,0] + back_o_compliance[k,0] > sum(purchase[i,k] for i in inst_gen.Suppliers)), \
                    f'Demand/backlogs compliance with purchased items of product {k} exceed the purchase'

                assert not sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) > inst_gen.W_d[self.t][k], \
                    f'Trying to comply a non-existing demand of product {k}' 
                
                for o in range(1, self.O_k[k] + 1):
                    assert not (inst_gen.other_params['backorders'] != 'backlogs' and demand_compliance[k,o] > self.state[k,o]), \
                        f'Demand compliance with inventory items exceed the stored items  ({k},{o})' 
                    
                    assert not (inst_gen.other_params['backorders'] == 'backlogs' and demand_compliance[k,o] + back_o_compliance[k,o] > self.state[k,o]), \
                        f'Demand/Backlogs compliance with inventory items exceed the stored items ({k},{o})'

            # backlogs
            if self.other_params['backorders'] == 'backlogs':
                for k in inst_gen.Products:
                    assert not sum(back_o_compliance[k,o] for o in range(self.O_k[k])) > self.state[k,'B'], \
                        f'Trying to comply a non-existing backlog of product {k}'
            
            elif self.others['backorders'] == False:
                for k in inst_gen.Products:
                    assert not sum(demand_compliance[k,o] for o in range(self.O_k[k] + 1)) < inst_gen.W_d[self.t][k], \
                        f'Demand of product {k} was not fulfilled'


    # Generates empty dicts 
    def generate_empty_inv_action(self, inst_gen: instance_generator) -> tuple[dict,dict]:
        purchase = {(i,k):0 for i in inst_gen.Suppliers for k in inst_gen.Products}
        demand_compliance = {(k,o):0 for k in inst_gen.Products for o in [0]+self.Ages[k]}

        return purchase, demand_compliance


    # Simple function to visualize the inventory
    def print_inventory(self, inst_gen: instance_generator) -> None:
        max_O = max([self.O_k[k] for k in inst_gen.Products])
        listamax = [[self.state[k,o] for o in self.Ages[k]] for k in inst_gen.Products]
        df = pd.DataFrame(listamax, index=pd.Index([str(k) for k in inst_gen.Products], name='Products'),
        columns=pd.Index([str(o) for o in range(1, max_O + 1)], name='Ages'))

        print(df)


    # Simple function to print state and main parameters
    def print_state(self, inst_gen: instance_generator) -> None:
        print(f'################################### STEP {self.t} ###################################')
        print('INVENTORY')
        max_age = max(list(self.O_k.values()))
        string = 'K \ O \t '
        for o in range(1, max_age + 1):     string += f'{o} \t'
        print(string)
        for k in inst_gen.Products:
            string = f'k{k} \t '
            for o in self.Ages[k]:  string += f'{self.state[k,o]} \t'
            print(string)
        
        print('\n')
        print('DEMAND')
        string1 = 'K';  string2 = 'd'
        for k in inst_gen.Products: string1 += f'\t{k}';    string2 += f'\t{inst_gen.W_d[self.t][k]}'
        print(string1)
        print(string2)

        print('\n')
        print('AVAILABLE QUANTITIES')
        string = 'M\K \t'
        for k in inst_gen.Products:     string += f'{k} \t'
        print(string)
        for i in inst_gen.Suppliers:
            new_string = f'{i}\t'
            for k in inst_gen.Products:
                if inst_gen.W_q[self.t][i,k] == 0:  new_string += f'{inst_gen.W_q[self.t][i,k]}\t'
                else:       new_string += f'{inst_gen.W_q[self.t][i,k]}\t'
            print(new_string)

        print('\n')


    # Simple function to print an action
    def print_action(self, action, inst_gen: instance_generator) -> None:
        if self.config['routing']:
            del action[0]

        if self.config['inventory']:
            print(f'####################### Action {self.t} #######################')
            print('Purchase')
            string = 'M\K \t'
            for k in inst_gen.Products:     string += f'{k} \t'
            print(string)
            for i in inst_gen.Suppliers:
                new_string = f'{i}\t'
                for k in inst_gen.Products:
                    new_string += f'{action[0][i,k]}\t'
                print(new_string)
            print('\n')

            print('Demand Compliance')
            max_age = max(list(self.O_k.values()))
            string = 'K \ O \t '
            for o in range(1, max_age + 1):     string += f'{o} \t'
            print(string)
            for k in inst_gen.Products:
                string = f'k{k} \t '
                for o in [0]+self.Ages[k]:  string += f'{action[1][k,o]} \t'
                print(string)

            print('\n')


    # Printing a representation of the environment (repr(env))
    def __repr__(self):
        return f'Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance.'