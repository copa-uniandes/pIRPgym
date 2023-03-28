from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policies
import matplotlib.pyplot as plt; from matplotlib.gridspec import GridSpec; from matplotlib.transforms import Affine2D
import scipy.stats as st; import imageio; import time; from IPython.display import Image
from random import randint
import ast


### Instance generator
# SD-IB-IRP-PP model's parameters
backorders = 'backorders'                                       # Feature's parameters
stochastic_params = ['q','d']

look_ahead = ['q','d']                                          
historical_data = ['*']

env_config = { 'M': 10, 'K': 3, 'T': 7,  'F': 4,
            'S': 3,  'LA_horizon': 4, 'back_o_cost': 2000}      # Other parameters

q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [2,0.5]}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = randint(0,int(2e7))                             # Random seeds
det_rd_seed = randint(0,int(2e7))

inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, env_config = env_config)
inst_gen.generate_instance(det_rd_seed, stoch_rd_seed, q_params = q_params, p_params = p_params, d_params = d_params, h_params = h_params)


### Environment
routing = True
inventory = True
perishability = 'ages'
env = steroid_IRP(routing, inventory, perishability)

state, _ = env.reset(inst_gen, return_state = True)





'''
Policy Evaluation Fucntion
'''
def Policy_evaluation(inst_gen, det_rd_seed, stoch_rd_seed, stoch = True):  
    

    # Episode's and performance storage
    rewards = {};   states = {};   real_actions = {};   backorders = {};   la_decisions = {}
    realized_dem = {};   q_sample = {};   tws = {}; perished = {}; actions={}; times = {}

    # Generating environment and policies generator
    

    policy = policies()
    
    
    run_time = time.time()

    
    
    done = False
    while not done:
        #print_state(env)
        # Environment transition
        states[env.t] = state
        q_sample[env.t] = [_["sample_paths"]["q"][0,s] for s in env.Samples]
        realized_dem[env.t] = env.W_t["d"]

        # Transition
        if stoch: action, la_dec = policy.Stochastic_Rolling_Horizon(state, _, env)
        else: action, la_dec = policy.Myopic_Heuristic(state, _, env)

        if done or not stoch:    tws[env.t] = 1
        else:    tws[env.t] = _["sample_path_window_size"]
        state, reward, done, real_action, _,  = env.step(action)
        if done:    states[env.t] = state
        #print(env.t)
        #print_extras(env, real_action, _)
        
        # Data storage
        actions[ env.t-1] = action
        real_actions[env.t-1] = real_action
        backorders[env.t-1] = _["backorders"]
        perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in env.Products}
        rewards[env.t] = reward
        la_decisions[env.t-1] = la_dec

    times = time.time() - run_time
    iterables = (env.Suppliers, env.Products, env.Samples, env.M_kt, env.O_k, env.Horizon)
    costs = (env.c, env.h_t, env.p_t, env.back_o_cost)

    return (rewards, states, real_actions, backorders, la_decisions, tws, iterables, costs, perished, realized_dem, q_sample, actions, times, (det_rd_seed, stoch_rd_seed))


'''
Run a complete instance
'''
def run_instance(K, S, T, num_episodes):
    


    stochastic_policy = {}; myopic_policy = {}
    ep = 0
    
    # while ep < num_episodes:
    #     try:
    

    
    myopic_policy[ep] = Policy_evaluation(inst_gen, det_rd_seed = det_rd_seed, stoch_rd_seed = stoch_rd_seed, stoch=False)
        #     #stochastic_policy[ep] = Policy_evaluation(inst_gen, det_rd_seed = det_rd_seed, stoch_rd_seed = stoch_rd_seed, stoch=True)
        #     print(f"Done {ep}")
        #     ep += 1
        # except:
        #     print("Error")
        #     ep += 1
        #     continue
    
    return stochastic_policy, myopic_policy


'''
Running functions
'''
M = 5; K = 10; S = 7; T = 7; num_episodes = 3
stochastic_policy, myopic_policy = run_instance(K, S, T, num_episodes)