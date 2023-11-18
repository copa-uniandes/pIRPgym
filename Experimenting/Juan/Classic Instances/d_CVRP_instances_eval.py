######################################     Modules     #######################################
import sys
from time import process_time
import os
import pickle

import verbose_module as verb
sys.path.append('../../.')
import pIRPgym


computer_name = input("Running experiment on mac? [Y/n]")
if computer_name == '': path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
else: path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'

policies = ['nn','RCL','GA','HGS','']


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
                'S':6,'LA_horizon':4,
                'd_max':2000,'hist_window':60,
                'back_o_cost':10000
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

for inst_set,inst_list in instances.items():
    for instance in inst_list:
        # Upload dCVRP instance
        purchase,benchmark = inst_gen.upload_CVRP_instance(inst_set,instance)