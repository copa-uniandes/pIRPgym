#%%
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator
from Visualizations import Routing_Visualizations


########################################## Instance generator ##########################################
# SD-IB-IRP-PP model's parameters
T = 7
M = 15
K = 10
F = 15

Q = 2000
d_max = 2000 

backorders = False              # Feature's parameters
stochastic_params = ['d','q']

look_ahead = ['d','q']
S = 6
LA_horizon = 3

historical_data = ['*']
hist_window = 40

env_config = {'M':M, 'K':K, 'T':T, 'F':F, 'Q':Q, 
              'S':S, 'LA_horizon':LA_horizon,
             'd_max':d_max, 'hist_window':hist_window}                    # Other parameters

# Creating instance generator object
inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)




#%%######################################### Random Instance ##########################################
# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [2,0.5]}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = 3                                               # Random seeds
det_rd_seed = 1

inst_gen.generate_basic_random_instance(det_rd_seed, stoch_rd_seed, q_params = q_params, p_params = p_params, d_params = d_params, h_params = h_params)

#%%######################################### CVRP Instance ##########################################
# CVRP Instance
# set = 'Li'
# instance = 'Li_21.vrp'
# # set = 'Golden'
# # instance = 'Golden_1.vrp'

# purchase = inst_gen.upload_CVRP_instance(set, instance)


#%%######################################### Environment ##########################################
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

# nn_routes, nn_distances = route_planner.Nearest_Neighbor(purchase, inst_gen)       # Nearest neighbor

# RCLc_routes, RCLc_distances = route_planner.RCL_constructive(purchase, inst_gen)  # RCL based constructive

# HyGeSe_routes, HyGeSe_distance = route_planner.HyGeSe(purchase, inst_gen)         # Hybrid Genetic Search (CVRP)

# MIP_routes, MIP_distance = route_planner.MIP_routing(purchase, inst_gen)          # Complete MIP

# CG_routes, CG_distance = route_planner.Column_Generation(purchase, inst_gen)      # Column Generation algorithm


#%%######################################### Visualizations ##########################################
routes = nn_routes; distances = nn_distances

# Visualizations
product = 0
Routing_Visualizations.route_availability_per_product(routes[1], product, inst_gen, env, True)

#%%
Routing_Visualizations.route_total_availability(routes[1], inst_gen, env)

#%%
product = 0
Routing_Visualizations.routes_availability_per_product(routes, product, inst_gen, env)

#%%
Routing_Visualizations.routes_total_availability(routes, inst_gen, env)



#%%######################################### Step ##########################################




























#%% Routing d-CVRP instance  
set = 'Li'
instance = 'Li_21.vrp'
# set = 'Golden'
# instance = 'Golden_1.vrp'

purchase = inst_gen.CVRP_instance(set, instance)


#%% 
### Step
# Policies
nn_routes, nn_distance = policy_generator.Routing.nearest_neighbor(purchase, inst_gen)      #  Nearest neighbor
#HyGeSe_routes, HyGeSe_distance = policy_generator.Routing.HyGeSe(purchase, inst_gen)       # Hybrid Genetic Search
MIP_routes, MIP_distance = policy_generator.Routing.MIP_routing(purchase, inst_gen)



# Call step function, transition
action = [nn_routes, purchase]
state, reward, done, real_action, _ = env.step(action, inst_gen, False)

#%%
