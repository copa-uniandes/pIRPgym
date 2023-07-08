#%%
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator
from Visualizations import RoutingV

########################################## Instance generator ##########################################
# SD-IB-IRP-PP model's parameters
T = 7
M = 10
K = 6
F = 4

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
backorders = False
env_config = {'M':M, 'K':K, 'T':T, 'F':F, 'Q':Q, 
              'S':S, 'LA_horizon':LA_horizon,
             'd_max':d_max, 'hist_window':hist_window}                    

# Creating instance generator object
inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, 
                              backorders, demand_type, env_config = env_config)



######################################### CundiBoy Instance ##########################################
# CundiBoy Instance

# Random seeds
det_rd_seed = 1
stoch_rd_seed = 3                                               

# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': 15}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': 6}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

inst_gen.generate_CundiBoy_instance(det_rd_seed,stoch_rd_seed,0,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params)


######################################### Environment ##########################################
# Environment
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = steroid_IRP(routing, inventory, perishability)

# Reseting the environment
state = env.reset(inst_gen, return_state = True)


#%%######################################### Diverse Routing Strategies ##########################################
purchase = policy_generator.Purchasing.avg_purchase_all(inst_gen, env)

# Routing Policies
route_planner = policy_generator.Routing

nn_routes, nn_distances, nn_loads, nn_time = route_planner.Nearest_Neighbor.NN_routing(purchase, inst_gen)                      # Nearest Neighbor
RCLc_routes, _, RCLc_distances, RCLc_loads, RCLc_time  = route_planner.RCL_constructive.RCL_routing(purchase, inst_gen)         # RCL based constructive
GA_routes, GA_distances, GA_loads, GA_time  = route_planner.GA.GA_routing(purchase, inst_gen, rd_seed=0, time_limit=30)         # Genetic Algorithm
# HyGeSe_routes, HyGeSe_distance, HyGeSe_time  = route_planner.HyGeSe.HyGeSe_routing(purchase, inst_gen)                        # Hybrid Genetic Search (CVRP)
# MIP_routes, MIP_distances, MIP_loads, MIP_time = route_planner.MIP.MIP_routing(purchase, inst_gen)                                                        # Complete MIP

# CG_routes, CG_distance = route_planner.Column_Generation(purchase, inst_gen)      # Column Generation algorithm
