"""
@author: Juan Betancourt
"""
################################## Modules ##################################
### Basic Librarires
from copy import copy, deepcopy
import pandas as pd
from typing import Union

### Instance generator
from .InstanceGenerator import instance_generator

### Building blocks
from .BuildingBlocks import Routing_management, Inventory_management, Environmental_management

################################ Description ################################
'''
        State (S_t): The state according to Powell (three components): 
            - Physical State (R_t):
                state:  Current available inventory (!*): (dict)  Inventory of product k in K of age o in O_k
                        When backlogsb are activated, will appear under age 'B'
            - Other deterministic info (Z_t):
                p: Prices: (dict) Price of product k in K at supplier i in M
                q: Available quantities: (dict) Available quantity of product k in K at supplier i in M
                h: Holding cost: (dict) Holding cost of product k in K
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
                - purchase: (dict) Units to purchase of product k in K at supplier i in M
                - demand_compliance: (dict) Units of product k in K of age o in O_k used to satisfy the demand 
                - backlogs_compliance: (dict) Units of product k in K of age o in O_k used to satisfy the backlogs


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
class steroid_IRP(): 
    
    # Initialization method
    def __init__(self,routing:bool=True,inventory:bool=True,perishability:Union[str,bool]=True):
        assert inventory >= bool(perishability),'Perishability only available with Inventory Problem'
        self.config = {'routing':routing,'inventory':inventory,'perishability':perishability}
        

    # Reseting the environment
    def reset(self,inst_gen:instance_generator,return_state:bool=False,strong_rate:bool=True):
        '''
        Reseting the environment. Generate or upload the instance.
        
        PARAMETERS:
        inst_gen: instange_generator object
        return_state: Indicates whether the state is returned
        '''  
        # EDITABLE The return statement can be set to return any of the parameters, historics, 
        # sample paths, etc. of the instance_generator element. 
        self.t = 0

        self.strong_rate = strong_rate

        if self.config['inventory']:
            self.state = Inventory_management.perish_per_age_inv.reset(inst_gen)
        
        if inst_gen.sustainability:
            self.payoff_matrix = {e:dict() for e in inst_gen.E+["costs"]}
            self.norm_matrix = {e:dict() for e in inst_gen.E+["costs"]}

        if return_state:
            return self.state
        
        


    # Step 
    def step(self,action:dict,inst_gen:instance_generator,validate_action:bool = False, warnings:bool = False, aggregated:bool = True, environmental:bool = False, verbose:bool = False):
        '''
        
        '''

        # -----------------------------
        # Actions Validation
        # -----------------------------

        if validate_action:
            self.action_validity(action, inst_gen)

        # -----------------------------
        # Real Actions Computing
        # -----------------------------

        # Action assembly
        real_action = dict()
        
        if inst_gen.s_params != False:
            '''
            When some parameters are stochastic, the chosen action might not be feasible. Therefore, an aditional intra-step 
            computation must be made and adjustments on the action might be necessary
            '''
            
            real_action["purchase"] = {(i,k): min(action['purchase'][i,k], inst_gen.W_q[self.t][i,k]) for i in inst_gen.Suppliers for k in inst_gen.Products}

            if environmental: real_action["demand_compliance"] = Inventory_management.perish_per_age_inv.get_real_dem_compl_FIFO(inst_gen,self,real_action["purchase"])      
            else: real_action["demand_compliance"] = Inventory_management.perish_per_age_inv.get_costs_dem_compl_without_waste(inst_gen, self, real_action["purchase"])

            if self.config['routing']:
                real_action["routing"] = action['routing']
        
        else:
            real_action["purchase"] = action["purchase"]
            real_action["demand_compliance"]  = action['demand_compliance']

            if self.config['routing']:
                real_action["routing"] = action['routing']

        if inst_gen.other_params['backorders'] == 'backlogs':
            real_back_o_compliance = action[-1]

        s_tprime, back_orders, perished = Inventory_management.perish_per_age_inv.update_inventory(inst_gen, self, real_action["purchase"], real_action["demand_compliance"], warnings)
        
        
        # -----------------------------
        # Rewards Computing
        # -----------------------------
        
        reward = dict()
        if self.config['routing']:
            reward['transport cost'] = Routing_management.price_routes(inst_gen,real_action["routing"],real_action["purchase"],aggregated = aggregated)
        if self.config['inventory']:
            reward['purchase cost'], reward['holding cost'], reward['backorders cost'] = Inventory_management.perish_per_age_inv.compute_costs(inst_gen, self, real_action["purchase"], s_tprime, back_orders, aggregated = aggregated)
            if inst_gen.sustainability:
                reward.update(Environmental_management.compute_environmental_impact(inst_gen, real_action["purchase"], real_action["routing"], s_tprime, perished, aggregated = aggregated))
        
        # -----------------------------
        # Time step
        # -----------------------------

        # Time step update and termination check
        self.t += 1
        done = self.check_termination(inst_gen)
        _ = (back_orders, perished)

        # State update
        self.state = s_tprime
        return self.state, reward, done, real_action, _


    # Checking for episode's termination
    def check_termination(self,inst_gen:instance_generator) -> bool:
        done = self.t >= inst_gen.T
        return done


    # Method to evaluate actions
    def action_validity(self,action,inst_gen:instance_generator):
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

                assert not sum(demand_compliance[k,o] for o in range(inst_gen.O_k[k] + 1)) > inst_gen.W_d[self.t][k], \
                    f'Trying to comply a non-existing demand of product {k}' 
                
                for o in range(1, inst_gen.O_k[k] + 1):
                    assert not (inst_gen.other_params['backorders'] != 'backlogs' and demand_compliance[k,o] > self.state[k,o]), \
                        f'Demand compliance with inventory items exceed the stored items  ({k},{o})' 

            
            if inst_gen.other_params['backorders'] == False:
                for k in inst_gen.Products:
                    assert not sum(demand_compliance[k,o] for o in range(inst_gen.O_k[k] + 1)) < inst_gen.W_d[self.t][k], \
                        f'Demand of product {k} was not fulfilled'


    # Printing a representation of the environment (repr(env))
    def __repr__(self):
        return f'Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products instance.'
    

    ''' Generates empty dicts '''
    def generate_empty_inv_action(self,inst_gen:instance_generator) -> tuple[dict,dict]:
        purchase = {(i,k):0 for i in inst_gen.Suppliers for k in inst_gen.Products}
        demand_compliance = {(k,o):0 for k in inst_gen.Products for o in [0]+inst_gen.Ages[k]}

        return purchase, demand_compliance



    ''' Visualize the inventory'''
    def print_inventory(self,inst_gen:instance_generator) -> None:
        max_O = max([inst_gen.O_k[k] for k in inst_gen.Products])
        listamax = [[self.state[k,o] for o in inst_gen.Ages[k]] for k in inst_gen.Products]
        df = pd.DataFrame(listamax, index=pd.Index([str(k) for k in inst_gen.Products], name='Products'),
        columns=pd.Index([str(o) for o in range(1, max_O + 1)], name='Ages'))

        print(df)


    ''' Print state and main parameters'''
    def print_state(self,inst_gen:instance_generator) -> None:
        print(f'################################### STEP {self.t} ###################################')
        print('INVENTORY')
        max_age = max(list(inst_gen.O_k.values()))
        string = 'K / O \t '
        for o in range(1, max_age + 1):     string += f'{o} \t'
        print(string)
        for k in inst_gen.Products:
            string = f'k{k} \t '
            for o in inst_gen.Ages[k]:  string += f'{self.state[k,o]} \t'
            print(string)
        
        print('\n')
        print('DEMAND')
        string1 = 'K';  string2 = 'd'
        for k in inst_gen.Products: string1 += f'\t{k}';    string2 += f'\t{inst_gen.W_d[self.t][k]}'
        print(string1)
        print(string2)

        print('\n')
        print('AVAILABLE QUANTITIES')
        string = 'M/K \t'
        for k in inst_gen.Products:     string += f'{k} \t'
        print(string)
        for i in inst_gen.Suppliers:
            new_string = f'{i}\t'
            for k in inst_gen.Products:
                if inst_gen.W_q[self.t][i,k] == 0:  new_string += f'{inst_gen.W_q[self.t][i,k]}\t'
                else:       new_string += f'{inst_gen.W_q[self.t][i,k]}\t'
            print(new_string)

        print('\n')


    ''' Print an action'''
    def print_action(self,action,inst_gen:instance_generator) -> None:
        if self.config['routing']:
            del action[0]

        if self.config['inventory']:
            print(f'####################### Action {self.t} #######################')
            print('Purchase')
            string = 'M/K \t'
            for k in inst_gen.Products:     string += f'{k} \t'
            print(string)
            for i in inst_gen.Suppliers:
                new_string = f'{i}\t'
                for k in inst_gen.Products:
                    new_string += f'{action[0][i,k]}\t'
                print(new_string)
            print('\n')

            print('Demand Compliance')
            max_age = max(list(inst_gen.O_k.values()))
            string = 'K / O \t '
            for o in range(1, max_age + 1):     string += f'{o} \t'
            print(string)
            for k in inst_gen.Products:
                string = f'k{k} \t '
                for o in [0]+inst_gen.Ages[k]:  string += f'{action[1][k,o]} \t'
                print(string)

            print('\n')
