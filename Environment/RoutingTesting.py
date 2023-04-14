#%%
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator

import matplotlib.pyplot as plt; from matplotlib.gridspec import GridSpec; from matplotlib.transforms import Affine2D
import scipy.stats as st; import imageio; import time; from IPython.display import Image
from random import seed, randint
import ast


### Instance generator
# SD-IB-IRP-PP model's parameters
backorders = False                                      # Feature's parameters
stochastic_params = False

look_ahead = False
historical_data = False

env_config = {}      # Other parameters


stoch_rd_seed = 0                                               # Random seeds
det_rd_seed = 1

# Creating instance generator object
inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)
purchase = inst_gen.Uchoa_CVRP_instance()

# inst_gen2 = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)
# purchase = inst_gen2.Uchoa_CVRP_instance('X-n1001-k43.vrp')

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
### Step
# Policies
routes, distance = policy_generator.Routing.nearest_neighbor(purchase, inst_gen)    #  Nearest neighbor

result = policy_generator.Routing.HyGeSe(purchase, inst_gen)                        # Hybrid Genetic Search
routes, distance = result.routes, result.cost


# Call step function, transition
action = [routes, purchase]
state, reward, done, real_action, _ = env.step(action, inst_gen, False)

#%%
