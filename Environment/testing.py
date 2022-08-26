from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policies
import matplotlib.pyplot as plt
import scipy.stats as st

#################################   Environment's parameters   #################################
# Random seed
rd_seed = 0

# SD-IB-IRP-PP model's parameters
backorderss = 'backorders'
stochastic_parameters = ['q','d']

# Feature's parameters
look_ahead = ['q','d']
historical_data = ['*']

# Action's parameters
validate_action = False
warnings = False

# Other parameters
num_episodes = 1
env_config = { 'M': 5, 'K': 5, 'T': 20,  'F': 1, 
               'S': 1,  'LA_horizon': 3, 'back_o_cost':1e12}

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
    q_sample = {}
    tws = {}
    env = steroid_IRP( look_ahead = look_ahead, 
                       historical_data = historical_data, 
                       backorders = backorderss,
                       stochastic_parameters = stochastic_parameters, 
                       env_config = env_config)

    policy = policies()

    for episode in range(num_episodes):

        state, _ = env.reset(return_state = True, rd_seed = rd_seed, 
          q_params = q_params, 
          p_params = p_params,
          d_params = d_params,
          h_params = h_params)
        done = False

        while not done:
            
            print(f'############################# {env.t} #############################')
            states[episode,env.t] = state
            action, la_dec = policy.stochastic_rolling_horizon(state, _, env)
            print(action[0])
            q_sample[episode,env.t] = [_["sample_paths"]["q"][0,s] for s in env.Samples]
            state, reward, done, real_action, _,  = env.step(action, validate_action = validate_action, warnings = warnings)

            real_actions[episode,env.t] = real_action
            backorders[episode,env.t] = _["backorders"]
            rewards[episode,env.t] = reward
            la_decisions[episode,env.t] = la_dec
            realized_dem[episode,env.t] = env.W_t["d"]
            if done:
                tws[episode,env.t] = 1
            else:
                tws[episode,env.t] = _["sample_path_window_size"]
            
    iterables = (env.Suppliers, env.Products, env.Samples, env.M_kt, env.O_k, env.Horizon)
    costs = (env.c, env.h_t, env.p_t, env.back_o_cost)

    return rewards, states, real_actions, backorders, la_decisions, realized_dem, q_sample, tws, iterables, costs


rewards, states, real_actions, backorders, la_decisions, realized_dem, q_sample, tws, iterables, costs = Policy_evaluation(num_episodes = num_episodes)

