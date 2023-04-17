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
backorders = 'backorders'                                       # Feature's parameters
stochastic_params = ['q','d']

look_ahead = ['q','d']
historical_data = ['*']

env_config = { 'M': 5, 'K': 10, 'T': 7,  'F': 4,
            'S': 8,  'LA_horizon': 4, 'back_o_cost': 2000}      # Other parameters

q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [2,0.5]}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = 0                                               # Random seeds
det_rd_seed = 1

# Creating instance generator object
inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)
inst_gen.generate_random_instance(det_rd_seed, stoch_rd_seed, q_params = q_params, p_params = p_params, d_params = d_params, h_params = h_params, discount=("strong","conc"))

### Environment
# Creating environment object
routing = False
inventory = True
perishability = 'ages'
env = steroid_IRP(routing, inventory, perishability)
policy = policy_generator()


#%%


from Policies import policies
policy2 = policies()

def Policy_evaluation(inst_gen):  
    

    # Episode's and performance storage
    rewards = {};   states = {};   real_actions = {};   backorders = {};   la_decisions = {}
    perished = {}; actions={}; #times = {}

    # Generating environment and policies generator
    
    run_time = time.time()

    state = env.reset(inst_gen, return_state = True)
    
    done = False
    while not done:
        #print_state(env)
        # Environment transition
        states[env.t] = state

        # Transition
        #print(f"Day {env.t}")
        action, la_dec = policy.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)

        state, reward, done, real_action, _,  = env.step(action[1:],inst_gen)
        if done:   states[env.t] = state
        
        # Data storage
        actions[env.t-1] = action
        real_actions[env.t-1] = real_action
        backorders[env.t-1] = _["backorders"]
        perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
        rewards[env.t] = reward
        la_decisions[env.t-1] = la_dec
    
    #times = time.time() - run_time

    return (rewards, states, real_actions, backorders, la_decisions, perished, actions)


(rewards, states, real_actions, backorders, la_decisions, perished, actions) = Policy_evaluation(inst_gen)



#%%
### Step
env.print_state(inst_gen)

# Generating action
purchase = policy_gen.Purchasing.det_purchase_all(env, inst_gen)
demand_compliance = policy_gen.Inventory.det_FIFO(state, purchase, env, inst_gen)

action = [purchase, demand_compliance]
env.print_action(action, inst_gen)

# Call step function, transition
state, reward, done, real_action, _ = env.step(action, inst_gen)

#%%
