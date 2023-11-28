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


experiments = [i for i in range(2,7)]
sizes = {1:5,2:10,3:15,4:20,5:40,6:60}

alphas = [0.1,0.2,0.4,0.6,0.8]
time_limits = [1,30,60,300,1800,3600]

init_times = {1:0.1,30:1,60:3,300:5,1800:5,3600:10}


def multiprocess_eval_stoch_policy( router,purchase,inst_gen,env, n=30,averages=True,
                                    dynamic_p=False,initial_seed=0,**kwargs):
    freeze_support()
    seeds = [i for i in range(initial_seed,initial_seed+n)]
    p = pool.Pool()

    def run_eval(seed):
        RCL_routes,RCL_obj,RCL_info,RCL_time = router(purchase,inst_gen,env.t,RCL_alphas=kwargs['RCL_alphas'],
                                                    adaptative=kwargs['adaptative'],rd_seed=seed,
                                                    time_limit=kwargs['time_limit'])
        return RCL_routes,RCL_obj,RCL_info,RCL_time                
    
    Results = p.map(run_eval,seeds)
    p.close()
    p.join()

    objectives = [i[1] for i in Results]
    vehicles = [len(i[0]) for i in Results]
    times = [i[3] for i in Results]
    
    return np.mean(objectives),round(np.mean(vehicles),2),np.mean(times),(np.median(objectives),
                np.std(objectives),np.min(objectives),np.max(objectives))



########################     Instance generator and Environment     #########################
### pIRP model's parameters
# Stochasticity
stochastic_params = ['d','q']
look_ahead = ['d','q']


# Historical data
historical_data = ['*']


# Other parameters
backorders = 'backorders'

# Random Instance
q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

d_params = {'dist': 'log-normal', 'r_f_params': [3,1]}          # Demand

h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

disc = ("strong","conc")


### Environment 
# Creating environment object
routing = True
inventory = True
perishability = 'ages'
env = pIRPgym.steroid_IRP(routing, inventory, perishability)



################################## Policy Evaluation ##################################
''' Parameters '''
verbose = False
start = process_time()
show_gap = True

cont = 10
for experiment in experiments:
    env_config = {'T':12,'Q':750,'S':2,'LA_horizon':2,
                  'd_max':2000,'hist_window':60,'back_o_cost':5000
                 }
    env_config['M'] = sizes[experiment]
    env_config['K'] = env_config['M']
    env_config['F'] = env_config['M']

    # Creating instance generator object
    inst_gen = pIRPgym.instance_generator(look_ahead, stochastic_params,
                                historical_data,backorders,env_config=env_config)
    
    for replica in range(1,6):
        if verbose: string = verb.CG_initialization.print_head(experiment,replica)

        instance_information = dict()

        inst_gen.generate_basic_random_instance(cont,cont+1,q_params=q_params,
                    p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)
        
        instance_information['inst_gen']=inst_gen
        cont+=2

        # Reseting the environment
        state = env.reset(inst_gen,return_state=True)
        done = False

        instance_information.update({'T':inst_gen.T,'M':list(),'Requirements':list()})
        results_information = {'NN':list(),'RCL':list(),'GA':list(),'CG':list()}
        results_information.update({f'CG_{time_limit}':list() for time_limit in time_limits})
        results_information.update({f'CG_{time_limit}_{alpha}':list() for time_limit in time_limits for alpha in alphas})

        while not done:
            print(f'\t {env.t}',end='\r')
            ''' Purchase '''
            try:
                [purchase,demand_compliance], la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
            except:
                purchase = pIRPgym.Purchasing.avg_purchase_all(inst_gen,env)
                demand_compliance = pIRPgym.Inventory.det_FIFO(purchase,inst_gen,env)
            pending_sup,requirements = pIRPgym.Routing.consolidate_purchase(purchase,inst_gen,env.t)
            instance_information['M'].append(len(pending_sup))
            instance_information['Requirements'].append(requirements)

            ''' Routing '''
            # Nearest Neighbor
            nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)
            results_information['NN'].append((nn_routes,nn_obj,nn_info,nn_time))

            # RCL Heuristic     
            if __name__=='__main__':
                RCL_obj,RCL_veh,RCL_time,(RCL_median,RCL_std,RCL_min,RCL_max) = multiprocess_eval_stoch_policy( pIRPgym.Routing.RCL_Heuristic,
                                                                                                purchase,inst_gen,env,n=30,
                                                                                                averages=True,dynamic_p=False,
                                                                                                time_limit=15,RCL_alphas=[0.05,0.1,0.2,0.35],
                                                                                                adaptative=True)
                results_information['RCL'].append((RCL_obj,RCL_veh,RCL_time,(RCL_median,RCL_std,RCL_min,RCL_max)))
            
            # # Genetic Algorithm
            # GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.GenticAlgorithm(purchase,inst_gen,env.t,return_top=False,
            #                                                                      rd_seed=0,time_limit=120,verbose=False)
            # results_information['GA'].append((nn_routes,nn_obj,nn_info,nn_time))

            # Column Generation
            for time_limit in time_limits:
                CG_routes,CG_obj,CG_info,CG_time,CG_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=time_limit,
                                                                                            verbose=False,return_num_cols=True)
                results_information[f'CG_{time_limit}'].append((CG_routes,CG_obj,CG_info,CG_time,CG_cols))

                for alpha in alphas:
                    CGi_routes,CGi_obj,CGi_info,CGi_time,CGi_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,
                                                                                                     time_limit=time_limit,verbose=False,
                                                                                                     heuristic_initialization=init_times[time_limit],
                                                                                                     return_num_cols=True,RCL_alpha=alpha)
                    results_information[f'CG_{time_limit}_{alpha}'].append((CGi_routes,CGi_obj,CGi_info,CGi_time,CGi_cols))

            ''' Compound action'''        
            action = {'routing':nn_routes,'purchase':purchase,'demand_compliance':demand_compliance}
            state,reward,done,real_action,_,  = env.step(action,inst_gen)
        

        save_pickle(experiment,replica,'instance_information',instance_information)
        save_pickle(experiment,replica,'NN',results_information['NN'])
        save_pickle(experiment,replica,'RCL',results_information['RCL'])
        for time_limit in time_limits:
            save_pickle(experiment,replica,f'CG_{time_limit}',results_information[f'CG_{time_limit}'])
            for alpha in alphas:
                save_pickle(experiment,replica,f'CG_{time_limit}_{alpha}',results_information[f'CG_{time_limit}_{alpha}'])

        print(f'✅ E{experiment}/R{replica}')

# %%
#
