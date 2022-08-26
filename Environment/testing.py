from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policies
import matplotlib.pyplot as plt
import scipy.stats as st

#################################   Environment's parameters   #################################
# Random seed
rd_seed = 0

# SD-IB-IRP-PP model's parameters
backorders = 'backorders'
stochastic_parameters = ['q','d']

# Feature's parameters
look_ahead = ['q','d']
historical_data = ['*']

# Action's parameters
validate_action = True
warnings = False

# Other parameters
num_episodes = 1
env_config = { 'M': 10, 'K': 10, 'T': 7,  'F': 4, 
               'S': 4,  'LA_horizon': 3}

q_params = {'distribution': 'c_uniform', 'min': 6, 'max': 20}
d_params = {'distribution': 'log-normal', 'mean': 2, 'stdev': 0.5}

p_params = {'distribution': 'd_uniform', 'min': 20, 'max': 60}
h_params = {'distribution': 'd_uniform', 'min': 20, 'max': 60}

#################################   Environment's parameters   #################################


'''

'''
def Policy_evaluation(num_episodes = 1000):
    
    rewards = {}
    states = {}
    real_actions = {}
    backorders = {}
    la_decisions = {}
    realized_dem = {}
    tws = {}
    env = steroid_IRP( look_ahead = look_ahead, 
                       historical_data = historical_data, 
                       backorders = backorders,
                       stochastic_parameters = stochastic_parameters, 
                       env_config = env_config)

    policy = policies()

    for episode in range(2):

        state, _ = env.reset(return_state = True, rd_seed = rd_seed, 
          q_params = q_params, 
          p_params = p_params,
          d_params = d_params,
          h_params = h_params)
        done = False

        while not done:
            
            states[episode,env.t] = state
            action, la_dec = policy.stochastic_rolling_horizon(state, _, env)
            print(state)
            print(action)
            state, reward, done, real_action, _,  = env.step(action, validate_action = validate_action, warnings = warnings)

            real_actions[episode,env.t] = real_action
            backorders[episode,env.t] = _["backorders"]
            rewards[episode,env.t] = reward
            la_decisions[episode,env.t] = la_dec
            realized_dem[episode,env.t] = env.W_t["d"]
            tws[episode,env.t] = _["sample_path_window_size"]
            
    iterables = (env.Suppliers,env.Products,env.Samples,env.M_kt,env.O_k,env.Horizon)
    costs = (env.c, env.h_t, env.p_t)

    return rewards, states, real_actions, backorders, la_decisions, realized_dem, tws, iterables, costs


rewards, states, real_actions, backorders, la_decisions, realized_dem, tws, iterables, costs = Policy_evaluation(num_episodes = num_episodes)

