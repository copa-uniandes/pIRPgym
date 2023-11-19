#%%
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator

import matplotlib.pyplot as plt; from matplotlib.gridspec import GridSpec; from matplotlib.transforms import Affine2D
import scipy.stats as st; import imageio; import time; from IPython.display import Image
from random import seed, randint
import ast

import numpy as np


#%%

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
        if inst_gen.other_params["demand_type"] == "aggregated":
            action, la_dec = policy.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
        else:
            action, la_dec = policy.Inventory.Stochastic_RH_Age_Demand(state,env,inst_gen)

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

    return [rewards, states, real_actions, backorders, la_decisions, perished, actions]


def run_instance(num_episodes, discount = ("strong","conc"), dem_dist = [2,0.5]):
    
    ''' Fixed Parameters '''
    
    backorders = 'backorders'                                       # Feature's parameters
    stochastic_params = ['q','d']

    look_ahead = ['q','d']
    historical_data = ['*']

    env_config = { 'M': 5, 'K': 10, 'T': 7,  'F': 4,
                'S': 10,  'LA_horizon': 4, 'back_o_cost': 2000}      # Other parameters

    q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
    p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

    h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

    ''' Demand Distribution and Price Discount '''

    d_params = {'dist': 'log-normal', 'r_f_params': dem_dist}
    disc = discount

    demand_type = "age"
    I0 = 5

    stoch_rd_seed = 0                                               # Random seeds
    det_rd_seed = 1

    policy1 = {}; policy2 = {}
    ep = 0
    det_rd_seed = randint(0,int(2e7))
    while ep < num_episodes:
        stoch_rd_seed = randint(0,int(2e7))

        inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, demand_type, env_config = env_config)
        inst_gen.generate_random_instance(det_rd_seed, stoch_rd_seed, I0, q_params = q_params, p_params = p_params, d_params = d_params, h_params = h_params, discount = disc)

        policy1[ep] = Policy_evaluation(inst_gen) + [inst_gen]
        #policy2[ep] = Policy_evaluation(inst_gen) + [inst_gen]
        print(f"Done episode {ep}")
        ep += 1
    
    return policy1, policy2

def plot_lognormal(mu,sigma):

    cols = ["goldenrod","blueviolet","salmon","deepskyblue"]

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,4))

    max_y = 0
    for i in range(len(mu)):
        s = np.random.lognormal(mu[i], sigma[i], 1000)
        #count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

        x = np.linspace(min(s), max(s), 10000)
        pdf = (np.exp(-(np.log(x) - mu[i])**2 / (2 * sigma[i]**2)) / (x * sigma[i] * np.sqrt(2 * np.pi)))

        if max(pdf) > max_y: max_y = max(pdf)

        ax.plot(x,pdf,linewidth=2,color=cols[i])
        ax.fill_between(x,pdf,color=cols[i],alpha=0.6)
    

    ax.set_ylim(0,max_y+0.03)



#%%



# Creating environment object
routing = False; inventory = True; perishability = 'ages'
env = steroid_IRP(routing, inventory, perishability)
policy = policy_generator()

policy1, policy2 = run_instance(num_episodes=5,dem_dist={0:[0.85,0.25], 1:[0.55,0.25], 2:[0.35,0.25], 3:[0.35,0.25]})


#%%


plot_lognormal([0.85,0.55,0.35,0.25],[0.25,0.25,0.25,0.25])



#%%





