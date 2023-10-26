#%%
import sys
from time import process_time

sys.path.append('../../.')
import pIRPgym
import pIRPgym

########################################## Instance generator ###########################################
# Instance Generator
# pIRP model's parameters
T = 7
M = 15
K = 10
F = 15

demand_type = 'aggregated'

# Vehicles
Q = 2000
d_max = 2000

# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']
S = 6
LA_horizon = 3 

# Historical data
historical_data = ['*']
hist_window = 40

# Other parameters
backorders = 'backorders'
back_o_cost = 10000  


env_config = {'M':M, 'K':K, 'T':T, 'F':F, 'Q':Q, 
              'S':S, 'LA_horizon':LA_horizon,
             'd_max':d_max, 'hist_window':hist_window,
             'back_o_cost':back_o_cost}

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)





######################################### Random Instance ##########################################
# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [3,1]}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = 0                                               # Random seeds
det_rd_seed = 1

inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params)

######################################### CVRP Instance ##########################################
# CVRP Instance
# set = 'Li'
# instance = 'Li_21.vrp'
# # set = 'Golden'
# # instance = 'Golden_1.vrp'

# purchase = inst_gen.upload_CVRP_instance(set, instance)


######################################### Environment ##########################################
# Environment
# Creating environment object
routing = True
inventory = True    
perishability = 'ages'
env = pIRPgym.steroid_IRP(routing, inventory, perishability)

# Reseting the environment
state = env.reset(inst_gen, return_state = True)


#%%######################################### Diverse Routing Strategies ##########################################
purchase = pIRPgym.Policies.Purchasing.avg_purchase_all(inst_gen, env)

nn_routes, nn_distances, nn_loads, nn_time = pIRPgym.Policies.Routing.NearestNeighbor(purchase,inst_gen,env.t)                      # Nearest Neighbor
RCLc_routes, _, RCLc_distances, RCLc_loads, RCLc_time  = pIRPgym.Policies.Routing.RCL_Heuristic(purchase,inst_gen,env.t)         # RCL based constructive
GA_routes, GA_distances, GA_loads, GA_time  = pIRPgym.Policies.Routing.HybridGenticAlgorithm(purchase, inst_gen,env.t,rd_seed=0,time_limit=30)         # Genetic Algorithm
HyGeSe_routes, HyGeSe_distance, HyGeSe_time  = pIRPgym.Policies.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t)                        # Hybrid Genetic Search (CVRP)
MIP_routes, MIP_distances, MIP_loads, MIP_time = pIRPgym.Policies.Routing.MixedIntegerProgram(purchase,inst_gen)                                                        # Complete MIP
CG_routes, CG_distances, CG_loads, CG_time =  pIRPgym.Policies.Routing.ColumnGeneration(purchase,inst_gen,env.t)      # Column Generation algorithm


#%%######################################### Visualizations ##########################################
# Routing strategies comparison
data = {
        'NN':[nn_routes,nn_distances, nn_loads, nn_time,0],
        'RCL':[RCLc_routes, RCLc_distances, RCLc_loads, RCLc_time,0],
        'GA':[GA_routes, GA_distances, GA_loads, GA_time,0],
        # 'HyGeSe':[HyGeSe_routes, HyGeSe_distance, HyGeSe_time,0],
        # 'MIP':[MIP_routes, MIP_distances, MIP_loads, MIP_time,0]
        # 'ColGen':[CG_routes, CG_distances, CG_loads, CG_time,0]
        }

pIRPgym.RoutingV.compare_routing_strategies(inst_gen, data)

#%% Routes analytics
routes = CG_routes; distances = CG_distances

# Visualizations
product = 0
pIRPgym.RoutingV.route_availability_per_product(routes[1], product, inst_gen, env, True)

#%%
pIRPgym.RoutingV.route_total_availability(routes[1], inst_gen, env)

#%%
product = 0
pIRPgym.RoutingV.routes_availability_per_product(routes, product, inst_gen, env)

#%%
pIRPgym.RoutingV.routes_total_availability(routes, inst_gen, env)



#%%######################################### Step ##########################################







