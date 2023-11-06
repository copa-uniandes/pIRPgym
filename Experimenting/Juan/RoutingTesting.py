#%%#########
import sys
from time import process_time

import verbose_module
sys.path.append('../../.')
import pIRPgym

########################################## Instance generator ###########################################
# Instance Generator
### pIRP model's parameters
# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']


# Historical data
historical_data = ['*']


# Other parameters
backorders = 'backorders'

env_config = {  'M':13,'K':15,'T':12,'F':13,'Q':2000,
                'S':6,'LA_horizon':4,
                'd_max':2000,'hist_window':60,
                'back_o_cost':10000
            }

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)




#%%######################################## Random Instance ##########################################
# Random Instance
# q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
# p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

# d_params = {'dist': 'log-normal', 'r_f_params': [3,1]}          # Demand

# h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

# stoch_rd_seed = 0                                               # Random seeds
# det_rd_seed = 1

# disc = ("strong","conc")

# inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
#                                         p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)

#%%######################################## CVRP Instance ##########################################
# CVRP Instance
# set = 'Li'
# instance = 'Li_21.vrp'
set = 'Golden'
instance = 'Golden_1.vrp'

purchase = inst_gen.upload_CVRP_instance(set, instance)
purchase = {key[0]:value for key,value in purchase.items()}


#%%######################################## Environment ##########################################
# Environment
# Creating environment object
routing = True
inventory = False    
perishability = False
env = pIRPgym.steroid_IRP(routing,inventory,perishability)
env.reset(inst_gen)

''' Parameters '''
verbose = True
instances = []

#%%######################################### Routing testing on classic instances ##########################################
for instance in instances:
    nn_routes, nn_obj, nn_info, nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)                                         # Nearest Neighbor
    if verbose: string = verbose_module.print_routing_update(string,nn_obj,len(nn_info[0]))
    RCLc_routes, _, RCLc_distances, RCLc_loads, RCLc_time  = pIRPgym.Routing.RCL_Heuristic(purchase,inst_gen,env.t)                                 # RCL based constructive
    if verbose: string = verbose_module.print_routing_update(string,sum(RCLc_distances),len(RCLc_routes))
    GA_routes,GA_distances,GA_loads,GA_time,_ = pIRPgym.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,return_top=False,rd_seed=0,time_limit=5)    # Genetic Algorithm
    if verbose: string = verbose_module.print_routing_update(string,sum(GA_distances),len(GA_routes))
    HyGeSe_routes, HyGeSe_distance, HyGeSe_time  = pIRPgym.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t,time_limit=5)                                   # Hybrid Genetic Search (CVRP)
    if verbose: string = verbose_module.print_routing_update(string,HyGeSe_distance,len(HyGeSe_routes))
    MIP_routes, MIP_obj, MIP_info, MIP_time = pIRPgym.Routing.MixedIntegerProgram(purchase,inst_gen,env.t)
    if verbose: string = verbose_module.print_routing_update(string,MIP_obj,len(MIP_info[0]))
    CG_routes, CG_distances, CG_loads, CG_time = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,verbose=False)       # Column Generation algorithm                  
    if verbose: string = verbose_module.print_routing_update(string,sum(CG_distances),len(CG_routes),end=True)                                        # Column Generation algorithm


#%%
import time

def simulate_episode():
    # Simulate some processing for the episode
    time.sleep(1)

def print_strategies_status(strategies):
    status_str = " ".join(f"{strategy}: {'✔' if is_over else ' '}" for strategy, is_over in strategies.items())
    print(status_str, end="\r")

# List of routing strategies
routing_strategies = ["A", "B", "C"]

# Simulate episodes
for episode in range(10):
    # Simulate processing for each time-step in the episode
    strategies_status = {strategy: episode >= 5 for strategy in routing_strategies}
    print_strategies_status(strategies_status)
    simulate_episode()

# Print a newline after completion
print()

#%%######################################### Visualizations ##########################################
# Routing strategies comparison
data = {
        'NN':[nn_routes,nn_distances, nn_loads, nn_time,0],
        'RCL':[RCLc_routes, RCLc_distances, RCLc_loads, RCLc_time,0],
        # 'GA':[GA_routes, GA_distances, GA_loads, GA_time,0],
        'HyGeSe':[HyGeSe_routes, HyGeSe_distance, HyGeSe_time,0],
        'MIP':[MIP_routes, MIP_distances, MIP_loads, MIP_time,0],
        # 'ColGen':[CG_routes, CG_distances, CG_loads, CG_time,0]
        }

pIRPgym.Visualizations.RoutingV.compare_routing_strategies(inst_gen,data)

#%% Routes analytics
routes = nn_routes; distances = nn_distances

# Visualizations
product = 0
pIRPgym.Visualizations.RoutingV.route_availability_per_product(routes[1], product, inst_gen, env, True)

#%%
pIRPgym.Visualizations.RoutingV.route_total_availability(routes[1], inst_gen, env)

#%%
product = 0
pIRPgym.Visualizations.RoutingV.routes_availability_per_product(routes, product, inst_gen, env)

#%%
pIRPgym.Visualizations.RoutingV.routes_total_availability(routes, inst_gen, env)



#%%######################################### Step ##########################################







