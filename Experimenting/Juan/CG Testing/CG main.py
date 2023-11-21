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
if computer_name == '': 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Column Generation/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Column Generation/'


def save_pickle(experiment,replica,policy,performance):
    with open(experiments_path+f'Experiment {experiment}/Replica {replica}/{policy}.pkl','wb') as file:
        # Use pickle.dump to serialize and save the dictionary to the file
        pickle.dump(performance,file)



experiments = [1,2,3,4,5,60]
sizes = {1:5,2:10,3:15,4:20,5:40,6:60}

alphas = [0.1,0.2,0.4,0.6]

init_times = {1:0.5,2:1,3:2,4:2,5:5,6:10}




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
verbose = True
start = process_time()
show_gap = True
string = str()

time_limit = 30

cont = 100
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
    
    for replica in range(1,4):
        if verbose: string = verb.CG_initialization.print_head(experiment,replica)

        instance_information = dict()

        inst_gen.generate_basic_random_instance(cont,cont+1,q_params=q_params,
                    p_params=p_params,d_params=d_params,h_params=h_params,discount=disc)
        instance_information['inst_gen']=inst_gen
        cont+=2

        # Reseting the environment
        state = env.reset(inst_gen,return_state=True)
        done = False

        instance_information.update({'M':list(),'Requirements':list(),'NN_routes':list(),'NN_obj':list()})
        results_information = {'CG':list()}
        results_information.update({f'CG_{alpha}':list() for alpha in alphas})

        while not done:
            ''' Purchase'''
            try:
                [purchase,demand_compliance], la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
            except:
                purchase = pIRPgym.Purchasing.avg_purchase_all(inst_gen,env)
                demand_compliance = pIRPgym.Inventory.det_FIFO(purchase,inst_gen,env)

            nn_routes,nn_obj,nn_info,nn_time = pIRPgym.Routing.NearestNeighbor(purchase,inst_gen,env.t)
            instance_information['NN_routes'].append(nn_routes)
            instance_information['NN_obj'].append(nn_obj)

            if verbose: string = verb.CG_initialization.print_step(env.t,purchase,len(nn_routes),nn_obj)
            
            ''' Routing '''
            CG_routes,CG_obj,CG_info,CG_time,CG_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=1800,
                                                                                        verbose=False,return_num_cols=True)       # Column Generation algorithm    
            results_information['CG'].append((CG_routes,CG_obj,CG_info,CG_time,CG_cols))
            
            if verbose: string = verb.CG_initialization.print_update(string,CG_time,CG_cols[1],len(CG_routes),CG_obj)

            for alpha in alphas:
                end = False
                if alpha == alphas[-1]:
                    end = True
                CGinit_routes,CGinit_obj,CGinit_info,CGinit_time,CGinit_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=False,
                                                                                                                verbose=False,heuristic_initialization=init_times[experiment],
                                                                                                                   return_num_cols=True,RCL_alpha=alpha)
                results_information[f'CG_{alpha}'].append((CGinit_routes,CGinit_obj,CGinit_info,CGinit_time,CGinit_cols))
                
                if verbose: string = verb.CG_initialization.print_update(string,CGinit_time,CGinit_cols,len(CGinit_routes),CGinit_obj,end=end)

            ''' Compound action'''        
            action = {'routing':nn_routes,'purchase':purchase,'demand_compliance':demand_compliance}

            state, reward, done, real_action, _,  = env.step(action,inst_gen)
        
        save_pickle(experiment,replica,'instance_information',instance_information)
        save_pickle(experiment,replica,f'CG',results_information[f'CG'])
        for alpha in alphas:
            save_pickle(experiment,replica,f'CG_{alpha}',results_information[f'CG_{alpha}'])

        print('\n')

