#%%######################################### Modules #########################################
import sys
from time import process_time
import os

import verbose_module
sys.path.append('../../.')
import pIRPgym

########################################## Instance generator ###########################################
# Instance Generator
### pIRP model's parameters
# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']


# Historical data
historical_data = ['*']


# Other parameters
backorders = 'backorders'

env_config = {  'M':13,'K':15,'T':12,'F':13,'Q':2000,
                'S':6,'LA_horizon':4,
                'd_max':2000,'hist_window':60,
                'back_o_cost':10000
            }

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)


######################################### Environment ##########################################
# Environment
# Creating environment object
routing = True
inventory = False    
perishability = False
env = pIRPgym.steroid_IRP(routing,inventory,perishability)
env.reset(inst_gen)

''' Parameters '''
verbose = True
show_gap = True
instances = dict()
instances['Li'] = [i for i in os.listdir('/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPenv/pIRPgym/Instances/CVRP Instances/dCVRP/Li') if i[-3:]=='vrp']
instances['Golden'] = [i for i in os.listdir('/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPenv/pIRPgym/Instances/CVRP Instances/dCVRP/Golden') if i[-3:]=='vrp']
instances['Li'].sort();instances['Golden'].sort()

#%%######################################### Routing testing on classic instances ##########################################
if verbose: verbose_module.routing_instances.print_head(show_gap)

for ss,inst_list in instances.items():
    if ss == 'Golden': print('\n')
    for instance in inst_list:
        # Upload dCVRP instance
        purchase,benchmark = inst_gen.upload_CVRP_instance(ss, instance)

        if verbose: string = verbose_module.routing_instances.print_inst(ss,instance,benchmark[0],benchmark[1])

        nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)                                         # Nearest Neighbor
        if verbose: string = verbose_module.routing_instances.print_routing_update(string,
                                                                    nn_obj,len(nn_routes),nn_time,show_gap,benchmark)
        RCL_routes,RCL_obj,RCL_info,RCL_time  = pIRPgym.Routing.RCL_Heuristic(purchase,inst_gen,env.t,RCL_alpha=0.001)                                 # RCL based constructive
        if verbose: string = verbose_module.routing_instances.print_routing_update(string,
                                                                    RCL_obj,len(RCL_routes),RCL_time,show_gap,benchmark)
        GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,return_top=False,rd_seed=0,time_limit=20)    # Genetic Algorithm
        if verbose: string = verbose_module.routing_instances.print_routing_update(string,
                                                                    GA_obj,len(GA_routes),GA_time,show_gap,benchmark)
        HyGeSe_routes, HyGeSe_distance, HyGeSe_time  = pIRPgym.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t,time_limit=20)                                   # Hybrid Genetic Search (CVRP)
        if verbose: string = verbose_module.routing_instances.print_routing_update(string,HyGeSe_distance,len(HyGeSe_routes),HyGeSe_time,show_gap,benchmark)
        CG_routes, CG_distances, CG_loads, CG_time = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,verbose=True)       # Column Generation algorithm                  
        if verbose: string = verbose_module.print_routing_update(string,sum(CG_distances),len(CG_routes),end=True)                                        # Column Generation algorithm



#%%######################################### Visualizations ##########################################


# pIRPgym.Visualizations.RoutingV.compare_routing_strategies(inst_gen,data)

# #%% Routes analytics
# routes = nn_routes; distances = nn_distances

# # Visualizations
# product = 0
# pIRPgym.Visualizations.RoutingV.route_availability_per_product(routes[1], product, inst_gen, env, True)

# #%%
# pIRPgym.Visualizations.RoutingV.route_total_availability(routes[1], inst_gen, env)

# #%%
# product = 0
# pIRPgym.Visualizations.RoutingV.routes_availability_per_product(routes, product, inst_gen, env)

# #%%
# pIRPgym.Visualizations.RoutingV.routes_total_availability(routes, inst_gen, env)

# %%
