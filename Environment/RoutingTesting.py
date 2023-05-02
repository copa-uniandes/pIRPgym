#%%
from InstanceGenerator import instance_generator, locations
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator

import matplotlib.pyplot as plt; from matplotlib.gridspec import GridSpec; from matplotlib.transforms import Affine2D
import scipy.stats as st; import imageio; import time; from IPython.display import Image
from random import seed, randint
import ast


### Instance generator
# SD-IB-IRP-PP model's parameters
T = 7
M = 20
K = 10
F = 30

backorders = False              # Feature's parameters
stochastic_params = ['d','q']

look_ahead = False

historical_data = ['*']
hist_window = 40

env_config = { 'M': M, 'K': K, 'T': T,  'F': F,
             'hist_window':hist_window}                    # Other parameters

# Creating instance generator object
inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)

# Random instance parameters
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [2,0.5]}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = 0                                               # Random seeds
det_rd_seed = 1

inst_gen.generate_basic_random_instance(det_rd_seed, stoch_rd_seed, q_params = q_params, p_params = p_params, d_params = d_params, h_params = h_params)

### Environment
# Creating environment object
routing = True
inventory = False
perishability = False
env = steroid_IRP(routing, inventory, perishability)

# Reseting the environment
env.reset(inst_gen)


### Policies
# Creating policy generator object
policy_gen = policy_generator()


#%%




































#%% Routing instance and generation 
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
