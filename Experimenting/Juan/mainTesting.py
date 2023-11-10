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

env_config = {'M':18,'K':22,'T':7,'F':18,'Q':2000,
              'S':6,'LA_horizon':4,
             'd_max':2000,'hist_window':60,
             'back_o_cost':100}

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)

##########################################    Random Instance    ##########################################
# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': (6,20)}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': (20,61)}

d_params = {'dist': 'log-normal', 'r_f_params': (3,1)}          # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': (20,61)}         # Holding costs

stoch_rd_seed = 1       # Random seeds
det_rd_seed = 1

disc = ("strong","conc")

inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)


# pIRPgym.Visualizations.InstanceV.plot_overlapping_distributions(q_params['r_f_params'],d_params['r_f_params'],type='line')

#########################################      Environment      ##########################################
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
verbose = True
strategies = ['CG','NN','RCL','GA']
start = process_time()
show_gap = True
string = str()

if verbose: string = verbose_module.routing_progress.print_iteration_head(strategies,show_gap)

# Episode's and performance storage
rewards=dict();  states=dict();   real_actions=dict();   backorders=dict();   la_decisions=dict()
perished=dict(); actions=dict()

indicators = ['Obj','time','vehicles','reactive_missing','extra_cost']
routing_performance = {s:{ind:list() for ind in indicators} for s in strategies}

# Reseting the environment
state = env.reset(inst_gen,return_state=True)

done = False
while not done:
    # Environment transition
    states[env.t] = state

    ''' Purchase'''
    [purchase,demand_compliance], la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)

    if verbose: string = verbose_module.routing_progress.print_step(env.t,start,purchase)
    
    ''' Routing '''
    # GA_extra_cost = env.compute_solution_real_cost(inst_gen,GA_routes,purchase)   
    if 'CG' in strategies:
        CG_routes,CG_obj,CG_info,CG_time = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=False,verbose=False)       # Column Generation algorithm                  
        extra_cost,total_missing = pIRPgym.Routing_management.evaluate_dynamic_potential(inst_gen,env,CG_routes,purchase)
        routing_performance['CG']['Obj'].append(CG_obj)
        routing_performance['CG']['time'].append(CG_time)
        routing_performance['CG']['vehicles'].append(len(CG_routes))
        routing_performance['CG']['reactive_missing'].append(total_missing)
        routing_performance['CG']['extra_cost'].append(extra_cost)

        if verbose: string = verbose_module.routing_progress.print_routing_update(string,CG_time,len(CG_routes),CG_obj)

    if 'NN' in strategies:
        nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)
        extra_cost,total_missing = pIRPgym.Routing_management.evaluate_dynamic_potential(inst_gen,env,nn_routes,purchase)
        routing_performance['NN']['Obj'].append(nn_obj)
        routing_performance['NN']['time'].append(nn_time)
        routing_performance['NN']['vehicles'].append(len(nn_routes))
        routing_performance['NN']['reactive_missing'].append(total_missing)
        routing_performance['NN']['extra_cost'].append(extra_cost)                                         # Nearest Neighbor
        if verbose and show_gap:   string = verbose_module.routing_progress.print_routing_update(string,nn_time,len(nn_routes),nn_obj,CG_obj=CG_obj)
        elif verbose and not show_gap:  string = verbose_module.routing_progress.print_routing_update(string,nn_time,len(nn_routes),nn_obj)

    if 'RCL' in strategies:
        RCL_obj,RCL_veh,RCL_time,RCL_EC,RCL_missing = pIRPgym.Routing.evaluate_stochastic_policy(pIRPgym.Routing.RCL_Heuristic,
                                                                                           purchase,inst_gen,env,n=30,return_average=True)
        routing_performance['RCL']['Obj'].append(RCL_obj)
        routing_performance['RCL']['time'].append(RCL_time)
        routing_performance['RCL']['vehicles'].append(RCL_veh)
        routing_performance['RCL']['reactive_missing'].append(RCL_missing)
        routing_performance['RCL']['extra_cost'].append(RCL_EC)
        # RCL_routes,RCL_obj,(RCL_distances,RCL_loads),RCL_time  = pIRPgym.Routing.RCL_Heuristic(purchase,inst_gen,env.t)                                 # RCL based constructive
        if verbose and show_gap: string = verbose_module.routing_progress.print_routing_update(string,RCL_time,RCL_veh,RCL_obj,CG_obj=CG_obj)
        elif verbose and not show_gap: string = verbose_module.routing_progress.print_routing_update(string,RCL_time,RCL_veh,RCL_obj)

    if 'GA' in strategies:
        GA_routes,GA_obj,(GA_distances,GA_loads),GA_time,_ = pIRPgym.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,return_top=False,rd_seed=0,time_limit=40)    # Genetic Algorithm
        extra_cost,total_missing = pIRPgym.Routing_management.evaluate_dynamic_potential(inst_gen,env,GA_routes,purchase)
        routing_performance['GA']['Obj'].append(GA_obj)
        routing_performance['GA']['time'].append(GA_time)
        routing_performance['GA']['vehicles'].append(len(GA_routes))
        routing_performance['GA']['reactive_missing'].append(total_missing)
        routing_performance['GA']['extra_cost'].append(extra_cost)
        if verbose and show_gap:  string = verbose_module.routing_progress.print_routing_update(string,GA_time,len(GA_routes),sum(GA_distances),CG_obj=CG_obj,end=True)
        elif verbose and not show_gap: string = verbose_module.routing_progress.print_routing_update(string,GA_time,len(GA_routes),sum(GA_distances),end=True)

    if 'HGS*' in strategies:
        HyGeSe_routes, HyGeSe_distance, HyGeSe_time  = pIRPgym.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t,time_limit=5)                                   # Hybrid Genetic Search (CVRP)
        extra_cost,total_missing = pIRPgym.Routing_management.evaluate_dynamic_potential(inst_gen,env,HyGeSe_routes,purchase)
        routing_performance['HGS*']['Obj'].append(HyGeSe_distance)
        routing_performance['HGS*']['time'].append(HyGeSe_time)
        routing_performance['HGS*']['vehicles'].append(len(HyGeSe_routes))
        routing_performance['HGS*']['reactive_missing'].append(total_missing)
        routing_performance['HGS*']['extra_cost'].append(extra_cost)
        if verbose and show_gap:  string = verbose_module.routing_progress.print_routing_update(string,HyGeSe_time,len(HyGeSe_routes),HyGeSe_distance,CG_obj=CG_obj)
        elif verbose and not show_gap:  string = verbose_module.routing_progress.print_routing_update(string,HyGeSe_time,len(HyGeSe_routes),HyGeSe_distance)


    ''' Compound action'''        
    action = {'routing':CG_routes,'purchase':purchase,'demand_compliance':demand_compliance}

    state, reward, done, real_action, _,  = env.step(action,inst_gen)
    if done:   states[env.t] = state
    
    # Data storage
    actions[env.t-1] = action
    real_actions[env.t-1] = real_action
    backorders[env.t-1] = _["backorders"]
    perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
    rewards[env.t] = reward 
    la_decisions[env.t-1] = la_dec

print('Finished episode!!!')

### Storing 

#%%####################################### Testing pricing algorithm  ######################################## 
# Generate random instance
inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)
# Reseting the environment
state = env.reset(inst_gen,return_state=True)
[purchase,demand_compliance], la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
CG_routes,CG_obj,CG_info,CG_time = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=False,verbose=False)       # Column Generation algorithm                  
nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)

testing_route = CG_routes[0]
reduced_cost = pIRPgym.Routing.PriceRoute(inst_gen,'canonic',testing_route,purchase,env.t)
print(f'The reduced cost is {reduced_cost}')




# %%
