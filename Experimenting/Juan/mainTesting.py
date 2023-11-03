#%%##########################################       Modules       ###########################################
# MODULES
import sys
from time import process_time

import verbose_module
sys.path.append('../../.')
import pIRPgym


###########################################  Instance Generator  ##########################################
# Instance Generator

### pIRP model's parameters
# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']


# Historical data
historical_data = ['*']


# Other parameters
backorders = 'backorders'

env_config = {'M':5,'K':15,'T':4,'F':15,'Q':2000,
              'S':6,'LA_horizon':3,
             'd_max':2000,'hist_window':60,
             'back_o_cost':10000}

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)

##########################################    Random Instance    ##########################################
# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [3,1]}          # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = 1       # Random seeds
det_rd_seed = 2

disc = ("strong","conc")

inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)


##########################################   CundiBoy Instance   ##########################################
# ### CundiBoy Instance
# # Random seeds
# det_rd_seed = 2
# stoch_rd_seed = 1                                        

# # Random Instance
# q_params = {'dist': 'c_uniform', 'r_f_params': 10}          # Offer
# p_params = {'dist': 'd_uniform', 'r_f_params': 0.3}

# d_params = {'dist': 'log-normal', 'r_f_params': 13}        # Demand

# h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

# I0 = 0

# inst_gen.generate_CundiBoy_instance(det_rd_seed,stoch_rd_seed,I0,q_params=q_params,p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)

# #%%#########################################     CVRP Instance     ##########################################
# # CVRP Instance
# set = 'Li'
# instance = 'Li_21.vrp'
# # set = 'Golden'
# # instance = 'Golden_1.vrp'

# purchase = inst_gen.upload_CVRP_instance(set, instance)

#%%#########################################      Environment      ##########################################
# Environment
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = pIRPgym.steroid_IRP(routing, inventory, perishability)

# Reseting the environment
state = env.reset(inst_gen,return_state=True)


#%%####################################### Single Episode/Singe Routing Policy Simulation  ########################################
# Episode simulation
# Simulations 
''' Parameters '''
num_episodes = 1
verbose = True
start = process_time()

# for episode in num_episodes:

if verbose: 
    string = verbose_module.print_iteration_head()

# Episode's and performance storage
rewards=dict();  states=dict();   real_actions=dict();   backorders=dict();   la_decisions=dict()
perished=dict(); actions=dict(); #times=dict() 

routing_performance = dict()
run_time = process_time()

# Reseting the environment
state = env.reset(inst_gen,return_state=True)

done = False
while not done:
    if verbose: 
        string = verbose_module.print_step(env.t,start)

    #print_state(env)
    # Environment transition
    states[env.t] = state 

    ''' Purchase'''
    [purchase,demand_compliance], la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
    string = verbose_module.print_purchase_update(string,inst_gen.W_p[env.t],purchase)

    ''' Routing '''
    # GA_extra_cost = env.compute_solution_real_cost(inst_gen,GA_routes,purchase)   

    nn_routes, nn_obj, nn_info, nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)                                         # Nearest Neighbor
    string = verbose_module.print_routing_update(string,nn_obj,len(nn_info[0]))
    RCLc_routes, _, RCLc_distances, RCLc_loads, RCLc_time  = pIRPgym.Routing.RCL_Heuristic(purchase,inst_gen,env.t)                                 # RCL based constructive
    string = verbose_module.print_routing_update(string,sum(RCLc_distances),len(RCLc_routes))
    GA_routes,GA_distances,GA_loads,GA_time,_ = pIRPgym.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,return_top=False,rd_seed=0,time_limit=5)    # Genetic Algorithm
    string = verbose_module.print_routing_update(string,sum(GA_distances),len(GA_routes))
    HyGeSe_routes, HyGeSe_distance, HyGeSe_time  = pIRPgym.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t)                                   # Hybrid Genetic Search (CVRP)
    string = verbose_module.print_routing_update(string,HyGeSe_distance,len(HyGeSe_routes))
    MIP_routes, MIP_distances, MIP_loads, MIP_time = pIRPgym.Routing.MixedIntegerProgram(purchase,inst_gen,env.t)
    string = verbose_module.print_routing_update(string,sum(MIP_distances),len(MIP_routes))
    # CG_routes, CG_distances, CG_loads, CG_time = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,verbose=False)       # Column Generation algorithm                  
    # string = verbose_module.print_routing_update(string,sum(CG_distances),len(CG_routes),end=True)

    ''' Compound action'''        
    action = {'routing':CG_routes, 'purchase':purchase, 'demand_compliance':demand_compliance}

    state, reward, done, real_action, _,  = env.step(action,inst_gen)
    if done:   states[env.t] = state
    
    # Data storage
    actions[env.t-1] = action
    real_actions[env.t-1] = real_action
    backorders[env.t-1] = _["backorders"]
    perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
    # rewards[env.t] = reward
    # la_decisions[env.t-1] = la_dec

print('Finished')



#%%####################################### Single Episode Simulation  ######################################## 



  
# %%

# %%
