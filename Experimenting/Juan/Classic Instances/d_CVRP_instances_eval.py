#%%#####################################     Modules     #######################################
import sys
from time import process_time
import os
import pickle

sys.path.append('../.')
import verbose_module as verb
sys.path.append('../../../.')
import pIRPgym


computer_name = input("Running experiment on mac? [Y/n]")
if computer_name == '': 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Classic Instances/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Classic Instances/'

policies1 = ['NN','RCL','HGS']
policies=dict()
policies['Li'] = policies1[:-1]
policies['Golden'] = policies1[:-1]
policies['Uchoa'] = policies1

def save_pickle(inst_set,policy,instance,performance):
    with open(experiments_path+f'{inst_set}/{policy}/{instance[:-4]}.pkl', 'wb') as file:
        # Use pickle.dump to serialize and save the dictionary to the file
        pickle.dump(performance, file)


########################     Instance generator and Environment     #########################
# Instance Generator
### pIRP model's parameters
# Stochasticity
stochastic_params = False
look_ahead = False


# Historical data
historical_data = False


# Other parameters
backorders = False

env_config = {  'M':13,'K':15,'T':12,'F':13,'Q':2000,
                'd_max':2000,
            }

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                              historical_data, backorders, env_config = env_config)

### Environment 
# Creating environment object
routing = True
inventory = False    
perishability = False
env = pIRPgym.steroid_IRP(routing,inventory,perishability)
env.reset(inst_gen)


instances = dict()
instances['Li'] = [i for i in os.listdir(path+'/pIRPgym/Instances/CVRP Instances/dCVRP/Li') if i[-3:]=='vrp']
instances['Golden'] = [i for i in os.listdir(path+'/pIRPgym/Instances/CVRP Instances/dCVRP/Golden') if i[-3:]=='vrp']
instances['Uchoa'] = [i for i in os.listdir(path+'pIRPgym/Instances/CVRP Instances/CVRP/Uchoa') if i[-3:]=='vrp']
instances['Li'].sort();instances['Golden'].sort()
instances['Uchoa'].sort();instances['Uchoa'] = instances['Uchoa'][1:] + [instances['Uchoa'][0]]


################################## Policy Evaluation ##################################
''' Parameters '''
verbose = True
show_gap = True

time_limit = 30

for inst_set,inst_list in instances.items():
    # if inst_set=='Li':continue
    if verbose: verb.routing_instances.print_head(policies[inst_set],inst_set,show_gap)
    for instance in inst_list:
        RCL_alphas = [0.01,0.05,0.1]
        if inst_set == 'Uchoa':
            RCL_alphas = [0.1,0.2,0.4,0.6]
        # Upload dCVRP instance
        purchase,benchmark = inst_gen.upload_CVRP_instance(inst_set,instance)

        if verbose: string = verb.routing_instances.print_inst(inst_set,instance,inst_gen.M,benchmark[1],benchmark[0])

        ''' Nearest Neighbor'''
        if 'NN' in policies[inst_set]:
            if 'NN'== policies[inst_set][-1]:end=True
            else:end=False
            nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)
            save_pickle(inst_set,'NN',instance,[nn_routes,nn_obj,nn_info,nn_time])          # Nearest Neighbor
            if verbose: string = verb.routing_instances.print_routing_update(string,nn_obj,len(nn_routes),
                                                                             nn_time,show_gap,benchmark,
                                                                             end=end)

        ''' RCL Heuristic'''
        if 'RCL' in policies[inst_set]:
            if 'RCL'== policies[inst_set][-1]:end=True
            else:end=False
            RCL_obj,RCL_veh,RCL_time,(RCL_std,RCL_min,RCL_max) = pIRPgym.Routing.\
                                                            evaluate_stochastic_policy( pIRPgym.Routing.RCL_Heuristic,
                                                                                        purchase,inst_gen,env,n=30,
                                                                                        averages=True,dynamic_p=False,
                                                                                        RCL_alphas=RCL_alphas)
            save_pickle(inst_set,'RCL',instance,[RCL_obj,RCL_veh,RCL_time,(RCL_std,RCL_min,RCL_max)])                 
            if verbose: string = verb.routing_instances.print_routing_update(string,RCL_obj,RCL_veh,RCL_time,
                                                                             show_gap,benchmark,end=end)

        ''' Genetic Algorithm '''
        if 'GA' in policies[inst_set]:
            if 'GA'== policies[inst_set][-1]:end=True
            else:end=False
            GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,
                                                                                       return_top=False,
                                                                                       rd_seed=0,time_limit=30)   
            save_pickle(inst_set,'GA',instance,[GA_routes,GA_obj,GA_info,GA_time])
            if verbose: string = verb.routing_instances.print_routing_update(string,GA_obj,len(GA_routes),GA_time,
                                                                             show_gap,benchmark,end=end)

        ''' Hybrid Genetic Search'''
        if 'HGS' in policies[inst_set]:
            HGS_routes,HGS_obj,HGS_time  = pIRPgym.Routing.HyGeSe.HyGeSe_routing(purchase,inst_gen,env.t,time_limit=30)
            save_pickle(inst_set,'HGS',instance,[HGS_routes,HGS_obj,HGS_time]) 
            if verbose: string = verb.routing_instances.print_routing_update(string,HGS_obj,len(HGS_routes),HGS_time,
                                                                             show_gap,benchmark,end=True)  
    
    print('\n')

# %%
