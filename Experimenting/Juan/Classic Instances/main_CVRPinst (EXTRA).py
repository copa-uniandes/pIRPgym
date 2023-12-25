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
if computer_name in ['','Y','y','Mac','mac']: 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Classic Instances/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Classic Instances/'

policies1 = ['GA']
policies=dict()
policies['Li'] = policies1[:-1]
policies['Golden'] = policies1[:-1]
policies['Uchoa'] = policies1

def save_pickle(inst_set,policy,instance,performance):
    with open(experiments_path+f'{inst_set}/{policy}/{instance[:-4]}.pkl','wb') as file:
        # Use pickle.dump to serialize and save the dictionary to the file
        pickle.dump(performance,file)


########################     Instance generator and Environment     #########################
# Instance Generator
### pIRP model's parameters
# Stochasticity
stochastic_params = False
look_ahead = False
historical_data = False
backorders = False

env_config = {  'M':13,'K':15,'T':12,'F':13,'Q':2000,
                'd_max':2000}

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                                      historical_data,backorders,env_config=env_config)

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
num_instances = sum(len(j) for j in instances.values())

################################## Policy Evaluation ##################################
''' Parameters '''
verbose = False
show_gap = True

# POPULATION_SIZES = [250,750,1000,2500]
# ELITE_PROPORTIONS = [0.05,0.15,0.3]
# MUTATION_RATES = [0.25,0.5,0.75]

for inst_set,inst_list in instances.items():
    if inst_set in ['Li','Golden']:continue
    if verbose: verb.routing_instances.print_head(policies[inst_set],inst_set,show_gap)
    RCL_alphas = [0.005,0.01,0.05,0.1]
    if inst_set == 'Uchoa':
        RCL_alphas = [0.01,0.05,0.2,0.35]
    for instance in inst_list[::-1]:
        sstr = f'{instance[:6]}\t'
        print(sstr,end='\r')
        # Upload dCVRP instance
        purchase,benchmark = inst_gen.upload_CVRP_instance(inst_set,instance)
        seed(inst_gen.M*2)

        if verbose: string = verb.routing_instances.print_inst(inst_set,instance,inst_gen.M,benchmark[1],benchmark[0])

        ''' Genetic Algorithm '''
        GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.GeneticAlgorithm(purchase,inst_gen,env.t,return_top=False,
                                                                                time_limit=600,Population_size=2500,
                                                                                Elite_prop=0.3,mutation_rate=0.25)   
        save_pickle(inst_set,'GA',instance,[GA_routes,GA_obj,GA_info,GA_time])
    
        sstr += f'✅  \t{round((GA_obj-benchmark[0])/benchmark[0],2)}'
        print(sstr)
    


# %%
