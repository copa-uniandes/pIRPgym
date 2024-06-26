#%%#####################################     Modules     #######################################
import sys
from time import process_time
import os
import pickle
import numpy as np
from numpy.random import seed

from multiprocess import pool,freeze_support

sys.path.append('../.')

import verbose_module as verb
sys.path.append('../../../.')
import pIRPgym


computer_name = input("Running experiment on mac? [Y/n]")
if computer_name == '': 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/First Phase/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/First Phase/'

# path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
# experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/First Phase/'

def save_pickle(experiment,replica,policy,performance):
    with open(experiments_path+f'Experiment {experiment}/Replica {replica}/{policy}.pkl','wb') as file:
        pickle.dump(performance,file)


Experiments = [i for i in range(1,6)]
Replicas = [i for i in range(1,6)]
sizes = {1:5,2:10,3:15,4:20,5:40,6:60}

alphas = [0.1,0.2,0.4,0.6,0.8]
time_limits = [1,30,60,300,1800,3600]

# init_times = {1:0.1,30:1,60:3,300:5,1800:5,3600:10}
init_times = {1:0.1,30:0.5,60:1,300:1,1800:2,3600:2}



for experiment in Experiments[::-1]:
    if experiment == 1: continue
    for replica in Replicas[::-1]:
        if experiment == 1 and replica in [1,2]:continue
        elif experiment == 5 and replica in [4,5]:continue
        CG_performance = {time_limit:list() for time_limit in time_limits}
        with open(experiments_path+f'Experiment {experiment}/Replica {replica}/instance_information.pkl','rb') as file:
            inst_info = pickle.load(file)

        inst_gen = inst_info['inst_gen']
        for t in range(len(inst_info['Requirements'])):
            requirements = inst_info['Requirements'][t]


            ''' Genetic Algorithm '''
            GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.GeneticAlgorithm(requirements,inst_gen,t,return_top=False,
                                                                                    time_limit=600,Population_size=2500,
                                                                                    Elite_prop=0.3,mutation_rate=0.5)
            save_pickle(experiment,replica,'GA',[GA_routes,GA_obj,GA_info,GA_time])
             
        #     for time_limit in time_limits:
        #         CG_routes,CG_obj,CG_info,CG_time,CG_cols = pIRPgym.Routing.ColumnGeneration(   requirements,inst_gen,t,
        #                                                                                             time_limit=time_limit,verbose=False,
        #                                                                                             heuristic_initialization=init_times[time_limit],
        #                                                                                             return_num_cols=True,RCL_alpha=0.9)
        #         CG_performance[time_limit].append((CG_routes,CG_obj,CG_info,CG_time,CG_cols))

        # for time_limit in time_limits:
        #     save_pickle(experiment,replica,f'CGv2_{time_limit}',CG_performance[time_limit])

        print(f'✅ E{experiment}/R{replica}')

# %%
#
