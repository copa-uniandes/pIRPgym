#%%##########################################       Modules       ###########################################
# MODULES
import sys
from time import process_time

sys.path.append('../../.')
import pIRPgym



#%%##########################################  Instance Generator  ##########################################
# Instance Generator
# pIRP model's parameters
T = 7
M = 15
K = 10
F = 15

demand_type = 'aggregated'

# Vehicles
Q = 2000
d_max = 2000

# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']
S = 6
LA_horizon = 3 

# Historical data
historical_data = ['*']
hist_window = 40

# Other parameters
backorders = 'backorders'
back_o_cost = 10000  


env_config = {'M':M, 'K':K, 'T':T, 'F':F, 'Q':Q, 
              'S':S, 'LA_horizon':LA_horizon,
             'd_max':d_max, 'hist_window':hist_window,
             'back_o_cost':back_o_cost}

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)

#%%#########################################    Random Instance    ##########################################
# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [3,1]}          # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

stoch_rd_seed = 0                                               # Random seeds
det_rd_seed = 1

disc = ("strong","conc")

inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)

#%%#########################################   CundiBoy Instance   ##########################################
### CundiBoy Instance
# Random seeds
det_rd_seed = 2
stoch_rd_seed = 1e3                                        

# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': 10}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': 0.3}

d_params = {'dist': 'log-normal', 'r_f_params': 13}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

I0 = 0

inst_gen.generate_CundiBoy_instance(det_rd_seed,stoch_rd_seed,I0,q_params=q_params,p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)

#%%#########################################     CVRP Instance     ##########################################
# CVRP Instance
set = 'Li'
instance = 'Li_21.vrp'
# set = 'Golden'
# instance = 'Golden_1.vrp'

purchase = inst_gen.upload_CVRP_instance(set, instance)

#%%#########################################      Environment      ##########################################
# Environment
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = pIRPgym.steroid_IRP(routing, inventory, perishability)

# Reseting the environment
state = env.reset(inst_gen, return_state = True)



#%%####################################### Single Episode/Singe Routing Policy Simulation  ########################################
# Episode simulation
# Simulations 
''' Simulations '''
num_episodes = 1


print(f'################### Episode simulation ##################')
# Episode's and performance storage
rewards=dict();  states=dict();   real_actions=dict();   backorders=dict();   la_decisions=dict()
perished=dict(); actions=dict(); #times=dict() 

routing_performance = dict()
run_time = process_time()

time_limit = 10

# Reseting the environment
state = env.reset(inst_gen,return_state=True)

done = False
while not done:
    print(f'-------------------- Step {env.t} --------------------')
    #print_state(env)
    # Environment transition
    states[env.t] = state 

    if inst_gen.other_params["demand_type"] == "aggregated":
        ''' Purchase'''
        [purchase,demand_compliance], la_dec = pIRPgym.Policies.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)

        ''' Routing '''
        nn_routes, nn_distances, nn_loads, nn_time = pIRPgym.Policies.Routing.NearestNeighbor(purchase,inst_gen,env.t)                      # Nearest Neighbor

        [GA_routes,GA_distances,GA_loads,GA_time],GA_top = pIRPgym.Policies.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,top=False,rd_seed=0,time_limit=time_limit);print('✅ GA routing')   # Genetic Algorithm
        GA_extra_cost = env.compute_solution_real_cost(inst_gen,GA_routes,purchase)                     

        ''' Compound action'''        
        action = {'routing':GA_routes, 'purchase':purchase, 'demand_compliance':demand_compliance}

    state, reward, done, real_action, _,  = env.step(action,inst_gen)
    if done:   states[env.t] = state
    
    # Data storage
    actions[env.t-1] = action
    real_actions[env.t-1] = real_action
    backorders[env.t-1] = _["backorders"]
    perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
    # rewards[env.t] = reward
    la_decisions[env.t-1] = la_dec

print('Finished')


#%%####################################### Single Episode Simulation  ######################################## 
