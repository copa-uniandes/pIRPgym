#%%#####################################     Modules     #######################################
import sys
from time import process_time
import os
import pickle
from numpy.random import seed

sys.path.append('../.')
import verbose_module as verb
sys.path.append('../../../.')
import pIRPgym


computer_name = input("Running experiment on mac? [Y/n]")
if computer_name == '': 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Column Generation/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Column Generation/'

def save_pickle(experiment,replica,policy,performance):
    with open(experiments_path+f'Experiment {experiment}/Replica {replica}/{policy}.pkl','wb') as file:
        # Use pickle.dump to serialize and save the dictionary to the file
        pickle.dump(performance,file)







########################     Instance generator and Environment     #########################
### pIRP model's parameters
# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']


# Historical data
historical_data = ['*']


# Other parameters
backorders = 'backorders'

# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [3,1]}          # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

disc = ("strong","conc")

env_config = {'T':5,'Q':750,'S':2,'LA_horizon':3,
                  'd_max':2000,'hist_window':30,'back_o_cost':5000
             }
env_config['M'] = 12
env_config['K'] = env_config['M']
env_config['F'] = env_config['M']


# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead,stochastic_params,
                            historical_data,backorders,env_config=env_config)
    


### Environment 
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = pIRPgym.steroid_IRP(routing, inventory, perishability)



#%%################################# Policy Evaluation ##################################
''' Parameters '''
verbose = True
start = process_time()
show_gap = True
string = str()

num_episodes = 50

routing_agent = pIRPgym.RoutingAgent(policies=['NN','RCL','CG'])

Q_table = {policy:1 for policy in routing_agent.policies}
N_table = {policy:1 for policy in routing_agent.policies}

res = {'baseline':list(),'random':list(),'Agent':list()}

import numpy as np
from numpy.random import random
### Q-Learning
# Parameter
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.8

start_epsilon_decaying = 1                  # First episode at which decay epsilon
end_epsilon_decaying = num_episodes - num_episodes // 4    # Last episode at which decay epsilon
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)     # Amount of decayment of epsilon         


print('----- Started -----')
cont = 100
for episode in range(num_episodes):
    seed(episode)
    print(f'Episode {episode}')

    cont += 2
    inst_gen.generate_basic_random_instance(cont,cont+1,q_params=q_params,
                    p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)
    state = env.reset(inst_gen,return_state=True)
    
    
    done = False
    ep_upper_bound = 0
    log = {'baseline':0,'random':0,'Agent':0}

    while not done: 
        try:
            [purchase,demand_compliance],la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
        except:
            purchase = pIRPgym.Purchasing.avg_purchase_all(inst_gen,env)
            demand_compliance = pIRPgym.Inventory.det_FIFO(purchase,inst_gen,env)

        _,requirements = pIRPgym.Routing.consolidate_purchase(purchase,inst_gen,env.t)
        DirShip_cost = routing_agent.direct_shipping_cost(requirements,inst_gen)
        ep_upper_bound+=DirShip_cost

        # Baseline policies
        baseline = routing_agent.policy_routing('NN',purchase,inst_gen,env.t)
        random_action = routing_agent.random_policy(purchase,inst_gen,env.t)

        # Select policy
        if random() > epsilon:
            best_router = min(Q_table,key=Q_table.get)
            best_action = routing_agent.policy_routing(best_router,purchase,inst_gen,env.t)
        else:
            best_action = random_action
            best_router = random_action[-1]

        # Update
        ratio = best_action[1]/DirShip_cost
        Q_table[best_router] = Q_table[best_router] + (1/N_table[best_router]) * (ratio-Q_table[best_router])
        N_table[best_router] += 1

        log['baseline'] += baseline[1]
        log['random'] += random_action[1]
        log['Agent'] += best_action[1]

        ''' Compound action'''       
        action = {'routing':baseline[0],'purchase':purchase,'demand_compliance':demand_compliance}

        new_state,reward,done,real_action,_, = env.step(action,inst_gen)
    
    res['baseline'].append(log['baseline'])
    res['random'].append(log['random'])
    res['Agent'].append(log['Agent'])

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:       # Decay epsilon
        epsilon -= epsilon_decay_value




#%%
results = {key:{'Cost':value} for key,value in res.items()}
pIRPgym.Visualizations.RoutingV.plot_indicator_evolution(results,'Cost',xlabel='Episode')


# %%





# %%



results = {key:{'Cost':value} for key,value in res.items()}


results = {'baseline': [39527,
  49884,
  55767,
  33854,
  31737,
  34550,
  31657,
  46227,
  33711,
  31309],
 'random': [35244.459400944135,
  41626.609547484964,
  43252.35894816159,
  29093.236836211046,
  27244.653500723507,
  33477,
  29866.555417379575,
  38015.70443759949,
  31028.36212378181,
  27078.944267260915],
 'Agent': [33966.82135308815,
  40175.42654355082,
  43017.3185920902,
  24704.7752956001,
  26088.93398017377,
  32140.549278172006,
  29622.77022927929,
  33811.403314651485,
  30863.709044118314,
  25398.21328770426]}
pIRPgym.Visualizations.RoutingV.plot_indicator_evolution(results,'Cost',xlabel='Episode')






# %%
