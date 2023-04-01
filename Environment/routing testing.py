#%%
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator, routing_blocks

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

env_config = { 'M': 3, 'T': 7,  'F': 4, 'Q': 40}      # Other parameters

stoch_rd_seed = 0                                               # Random seeds
det_rd_seed = 1
#%%
# Creating instance generator object
inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)
inst_gen.upload_Uchoa_CVRP_instance()

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
# generate empty purchase


# Call step function, transition
# state, reward, done, real_action, _ = env.step([routes], inst_gen)

#%%
