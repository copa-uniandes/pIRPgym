#%% Imports
''' Generate CundiBoy instance and environment '''  
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator
from Visualizations import RoutingV, InventoryV
from time import process_time; from numpy.random import randint; import sys

#%% Parameters

########################################## Instance generator ##########################################
# SD-IB-IRP-PP model's parameters
T = 7
M = 20
K = 15
F = 6

demand_type = 'aggregated'

# Vehicles
Q = 750
d_max = 4000

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


######################################### CundiBoy Instance ##########################################
### CundiBoy Instance
# Random seeds
det_rd_seed = 2                                          

# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': 15}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': 0.3}

d_params = {'dist': 'log-normal', 'r_f_params': 10}        # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

I0 = 0


######################################### Environment ##########################################
### Environment
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = steroid_IRP(routing, inventory, perishability)


#%% Simulations 
''' Simulations '''
num_episodes = 1
policy1 = dict(); policy2 = dict()
disc = ("strong","conc")

time_limit = 5

ep = 0
while ep < num_episodes:
    print(f'################### Episode {ep} ##################')
    stoch_rd_seed = (det_rd_seed+1)**2 + ep

    inst_gen = instance_generator(look_ahead,stochastic_params,historical_data,backorders,demand_type,env_config=env_config)
    inst_gen.generate_CundiBoy_instance(det_rd_seed,stoch_rd_seed,I0,q_params=q_params,p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)

    ################### policy_evaluation ###################
    # Episode's and performance storage
    rewards=dict();   states=dict();   real_actions=dict();   backorders=dict();   la_decisions=dict()
    perished=dict(); actions=dict(); #times=dict()

    routing_performance=dict()
    run_time = process_time()

    # Reseting the environment
    state = env.reset(inst_gen,return_state=True)
    
    done = False
    while not done:
        print(f'-------------------- Step {env.t} --------------------')
        #print_state(env)
        # Environment transition
        states[env.t] = state

        # Transition
        #print(f"Day {env.t}")
        if inst_gen.other_params["demand_type"] == "aggregated":
            ''' Purchase'''
            [purchase,demand_compliance], la_dec = policy_generator.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)

            ''' Routing '''
            #nn_routes,nn_distances,nn_loads,nn_time = policy_generator.Routing.Nearest_Neighbor.NN_routing(purchase,inst_gen,env.t);print('✅ NN routing')                       # Nearest Neighbor
            #nn_extra_cost = env.compute_solution_real_cost(inst_gen,nn_routes,purchase)
            RCLc_routes,_,RCLc_distances,RCLc_loads,RCLc_time = policy_generator.Routing.RCL_constructive.RCL_routing(purchase,inst_gen,env.t);print('✅ RCL routing')         # RCL based constructive
            RCLc_extra_cost = env.compute_solution_real_cost(inst_gen,RCLc_routes,purchase)
            #[GA_routes,GA_distances,GA_loads,GA_time],GA_top = policy_generator.Routing.GA.GA_routing(purchase,inst_gen,env.t,top=False,rd_seed=0,time_limit=time_limit);print('✅ GA routing')   # Genetic Algorithm
            #GA_extra_cost = env.compute_solution_real_cost(inst_gen,GA_routes,purchase)
            #HyGeSe_routes,HyGeSe_distance,HyGeSe_time = policy_generator.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t,time_limit=time_limit);print('✅ HGS routing')  # Hybrid Genetic Search (CVRP)
            #HyGeSe_extra_cost = env.compute_solution_real_cost(inst_gen,HyGeSe_routes,purchase)
            #MIP_routes, MIP_distances,MIP_loads, MIP_time = policy_generator.Routing.MIP.MIP_routing(purchase,inst_gen,env.t);print('✅ MIP routing')                          # Complete MIP
            #MIP_extra_cost = env.compute_solution_real_cost(inst_gen,MIP_routes,purchase)   
            # CG_routes,CG_distances,CG_loads,CG_time = policy_generator.Routing.CG_routing(purchase,inst_gen,env.t);print('✅ Column Generation routing')                                          # Column Generation Algorithm
            # CG_extra_cost = env.compute_solution_real_cost(inst_gen,CG_routes,purchase)                         

            data = {
                #'NN':[nn_routes,nn_distances,nn_loads,nn_time,nn_extra_cost],
                'RCL':[RCLc_routes,RCLc_distances, RCLc_loads,RCLc_time,RCLc_extra_cost],
                #'GA':[GA_routes,GA_distances,GA_loads,GA_time,GA_extra_cost,GA_top],
                #'CG':[MIP_routes,MIP_distances,MIP_loads,MIP_time,MIP_extra_cost],
                # 'ColGen':[CG_routes,CG_distances,CG_loads,CG_time,CG_extra_cost]
                }
            CG_routes = MIP_routes
            RoutingV.compare_routing_strategies(inst_gen, data)
            print('\n')
            
            action = [RCLc_routes, purchase, demand_compliance]

        else:
            action, la_dec = policy_generator.Inventory.Stochastic_RH_Age_Demand(state,env,inst_gen)

        state, reward, done, real_action, _,  = env.step(action,inst_gen)
        if done:   states[env.t] = state
        
        # Data storage
        routing_performance[env.t] = data

        actions[env.t-1] = action
        real_actions[env.t-1] = real_action
        backorders[env.t-1] = _["backorders"]
        perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
        # rewards[env.t] = reward
        la_decisions[env.t-1] = la_dec
    ################### policy_evaluation ###################

    policy1[ep] = [states,real_actions,backorders,la_decisions,perished,actions,inst_gen]
    # policy2[ep] = Policy_evaluation(inst_gen) + [inst_gen]

    print(f"Done episode {ep}")
    ep += 1

print('Finished')

# %%

#%% 
''' Routing Visualizations '''
for ep in routing_performance.keys():
    routing_performance[ep]['CG'][0] = [route for route in routing_performance[ep]['CG'][0] if route != [0,0]]
    routing_performance[ep]['CG'][0] = [route for route in routing_performance[ep]['CG'][0] if route != [0,0]]
    

RoutingV.render_routes(inst_gen,HyGeSe_routes)
RoutingV.plot_solutions(inst_gen,routing_performance)
RoutingV.plot_solutions_standarized(inst_gen,routing_performance)

#%%
ep = 1
# RoutingV.render_routes_diff_strategies(inst_gen,[routing_performance[ep]['GA'][0], routing_performance[ep]['CG'][0]])

env.t = 1
RoutingV.route_total_availability(MIP_routes[1], inst_gen, env)


#%% Plots

day = 3
k = inst_gen.Products[2]

ax_inv = InventoryV.inventories(day,k,inst_gen,real_actions,actions,la_decisions,states)




