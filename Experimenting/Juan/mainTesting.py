#%%##########################################       Modules       ###########################################
# MODULES
import sys
from time import process_time
import os
import pickle

import verbose_module as verb

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

env_config = {'M':30,'K':25,'T':10,'F':20,'Q':2000,
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


# inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
#                                         p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)






########################     Instance generator and Environment     #########################
path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
# path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)

instances = dict()
instances = [i for i in os.listdir(path+'pIRPgym/Instances/CVRP Instances/CVRP/Uchoa') if i[-3:]=='vrp']
instances.sort()
instances = instances[1:] + [instances[0]]

### Environment 
# Creating environment object
routing = True
inventory = False    
perishability = False
env = pIRPgym.steroid_IRP(routing,inventory,perishability)
env.reset(inst_gen)






################################## Policy Evaluation ##################################
''' Parameters '''
verbose = False
show_gap = True

# ranges = [i for i in range(1,11)]
# performance = {'nn':{i:{'Gap':list(),'Time':list()} for i in ranges},
#                'RCL':{i:{'Gap':list(),'Time':list()} for i in ranges},
#                'GA':{i:{'Gap':list(),'Time':list()} for i in ranges},
#                'HGS':{i:{'Gap':list(),'Time':list()} for i in ranges}}


if verbose: verb.routing_instances.print_head(show_gap)

for instance in instances[60:]:
    # Upload dCVRP instance
    purchase,benchmark = inst_gen.upload_CVRP_instance('Uchoa',instance)

    if verbose: string = verb.routing_instances.print_inst('Uchoa',instance,benchmark[0],benchmark[1])

    nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)                                         # Nearest Neighbor
    if verbose: string = verb.routing_instances.print_routing_update(string,
                                                                nn_obj,len(nn_routes),nn_time,show_gap,benchmark)
    # performance['nn'][len(purchase)//100]['Gap'].append((nn_obj-benchmark[0])/benchmark[0])
    # performance['nn'][len(purchase)//100]['Time'].append(nn_time)


    RCL_obj,RCL_veh,RCL_time = pIRPgym.Routing.evaluate_stochastic_policy(pIRPgym.Routing.RCL_Heuristic,
                                                                          purchase,inst_gen,env,n=30,averages=True,dynamic_p=False)
    if verbose: string = verb.routing_instances.print_routing_update(string,
                                                                RCL_obj,RCL_veh,RCL_time,show_gap,benchmark)
    # performance['RCL'][len(purchase)//100]['Gap'].append((RCL_obj-benchmark[0])/benchmark[0])
    # performance['RCL'][len(purchase)//100]['Time'].append(RCL_time)

    GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.HybridGenticAlgorithm(purchase,
                                                                               inst_gen,env.t,return_top=False,rd_seed=0,time_limit=30)    # Genetic Algorithm
    if verbose: string = verb.routing_instances.print_routing_update(string,
                                                                GA_obj,len(GA_routes),GA_time,show_gap,benchmark)
    # performance['GA'][len(purchase)//100]['Gap'].append((GA_obj-benchmark[0])/benchmark[0])
    # performance['GA'][len(purchase)//100]['Time'].append(GA_time)

    HGS_routes,HGS_obj,HGS_time  = pIRPgym.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t,time_limit=30)                                  # Hybrid Genetic Search (CVRP)
    if verbose: string = verb.routing_instances.print_routing_update(string,
                                                                    HGS_obj,len(HGS_routes),HGS_time,show_gap,benchmark,end=True)                                    # Column Generation algorithm
    # performance['HGS'][len(purchase)//100]['Gap'].append((HGS_distance-benchmark[0])/benchmark[0])
    # performance['HGS'][len(purchase)//100]['Time'].append(HGS_time)

    performance = { 'nn':{'Gap':(nn_obj-benchmark[0])/benchmark[0],'Time':nn_time},
                    'RCL':{'Gap':(RCL_obj-benchmark[0])/benchmark[0],'Time':RCL_time},
                    'GA':{'Gap':(GA_obj-benchmark[0])/benchmark[0],'Time':GA_time},
                    'HGS':{'Gap':(HGS_obj-benchmark[0])/benchmark[0],'Time':HGS_time}}
    
    # Open the file in binary write mode
    # a_file = open(path+f'Experimenting/Juan/Classic Instances/CVRP Results/{instance[:-4]}.pkl', 'wb')
    # pickle.dump(performance,a_file)
    # a_file.close()


    with open(path+f'Experimenting/Juan/Classic Instances/CVRP Results/{instance[:-4]}.pkl', 'wb') as file:
        # Use pickle.dump to serialize and save the dictionary to the file
        pickle.dump(performance, file)
    


#%%










#########################################      Environment      ##########################################
# Environment
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = pIRPgym.steroid_IRP(routing, inventory, perishability)

# # Reseting the environment
# state = env.reset(inst_gen,return_state=True)































#%%####################################### Testing pricing algorithm  ######################################## 
# Testing pricing algorithm
# Reseting the environment
state = env.reset(inst_gen,return_state=True)
done = False

res = {'nn':[0 for i in range(6)],'RCL':[0 for i in range(6)]}

while not done:
    print(f'step {env.t}')
    [purchase,demand_compliance], la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)    
    
    nn_routes,nn_obj,nn_info,nn_time,nn_r = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t,price_routes=True)
    for i in range(len(nn_routes)):
        res['nn'][i] += nn_r[i]; print(len(nn_routes))

    RCL_routes,RCL_obj,RCL_info,RCL_time,RCL_r = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t,price_routes=True)
    for i in range(len(RCL_routes)):
        res['RCL'][i] += RCL_r[i]; print(len(RCL_routes))

    ''' Compound action'''        
    action = {'routing':nn_routes,'purchase':purchase,'demand_compliance':demand_compliance}

    state, reward, done, real_action, _,  = env.step(action,inst_gen)
    





#%%
testing_route = nn_routes[3]
reduced_cost = pIRPgym.Routing.PriceRoute(inst_gen,testing_route,purchase,env.t,solution=CG_routes)
print(f'The reduced cost is {reduced_cost}')




#%% Comparing 













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

inst_gen.generate_basic_random_instance(det_rd_seed,stoch_rd_seed,q_params=q_params,
                                        p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)

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

