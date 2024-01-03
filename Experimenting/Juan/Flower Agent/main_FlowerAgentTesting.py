#%%##########################################       Modules       ###########################################
# MODULES
import sys
from time import process_time
import numpy as np;from numpy import random
import pickle
import datetime

sys.path.append('../../../.')
import pIRPgym

# computer_name = input("Running experiment on mac? [Y/n]")
# if computer_name == '': 
path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Flower Agent/'
# else: 
    # path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    # experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Flower Agent/'

#%%
# Environment
env = pIRPgym.steroid_IRP(True,True,True)

# Instance generator parameters
stochastic_params = ['d','q']   # Stochasticity
look_ahead = ['d','q']

historical_data = ['*']         # Historical data


# Other parameters
backorders = 'backorders'

p_params = {'dist':'d_uniform','r_f_params':(20,61)}

d_params = {'dist':'log-normal-trunc','r_f_params':(3,1),'break':80}         # Demand

h_params = {'dist':'d_uniform','r_f_params':(20,61)}         # Holding costs

env_config = {'T':7,'Q':1000,
              'S':3,'LA_horizon':3,
              'd_max':2500,'hist_window':60,
              'theta':0.6}


disc = ("strong","conc")

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead,stochastic_params,
                        historical_data,backorders,env_config=env_config)

testing_episodes = 10

Num_Episodes = [10,30,50]
Num_Suppliers = {10:[10,20,30,40,50],
                  30:[5,10,15,20,30],
                  50:[20,30]}

identifiers = {(10,10):0,(10,30):1,(10,50):2,(30,10):3,(30,30):4,(30,50):5,(50,10):6,(50,30):7,(50,50):8}


for episode in Num_Episodes:
    for size in Num_Suppliers:
        print(f'EXPERIMENT {identifiers[episode,size]}')
        results = {'info':(identifiers[episode,size],episode,size),'CG_results':list(),
                   'SL_results':list(),'cost_results':list()}

        with open(experiments_path+f'Training/M{size}-E{episode}.pkl', 'rb') as file:
            _,_,inst_gen,FlowerAgent = pickle.load(file)

        env_config['M']=size
        env_config['K']=env_config['M']
        env_config['F']=env_config['M']
        
        main_done = False
        ep_count = 0

        det_rd_seed = inst_gen.d_rd_seed             # Random seeds
        stoch_rd_seed = det_rd_seed*690
        results['seeds'] = [det_rd_seed,[stoch_rd_seed]]

        while not main_done:
            stoch_rd_seed+=1
            # try:
            inst_gen.generate_supplier_differentiated_random_instance(det_rd_seed,stoch_rd_seed,p_params=p_params,
                                                                        d_params=d_params,h_params=h_params,discount=disc)
            state = env.reset(inst_gen,return_state=True)
            done = False
            while not done:
                ''' Purchase '''
                [purchase,demand_compliance],la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
                total_purchase = sum(purchase.values())
                price_delta = pIRPgym.Routing_management.evaluate_purchase(inst_gen,purchase,env.t)


                ''' Generating routing solutions '''
                # Column Generation
                CG_routes,CG_obj,CG_info,CG_time,CG_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=1,
                                                                                            verbose=False,heuristic_initialization=0.2,
                                                                                            return_num_cols=True,RCL_alpha=0.8) 

                ''' Fitting routing to purchase '''
                SL_flower_info,cost_flower_info,solution_time = FlowerAgent.fit_purchase_to_flower(purchase,inst_gen,env.t,n=5)
                [SL_purchase,SL_demand_compliance],SL_la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen,fixed_suppliers=SL_flower_info[1])
                [cost_purchase,cost_demand_compliance],cost_la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen,fixed_suppliers=cost_flower_info[1])


                ''' Evaluating policies '''
                CG_missing,CG_r_missing,CG_extra_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,CG_routes,purchase,True)
                SL_missing,SL_r_missing,SL_extra_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,SL_flower_info[0],purchase,True)
                cost_missing,cost_r_missing,cost_extra_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,cost_flower_info[0],purchase,True)


                ''' Saving results'''
                CG_binary_coding = FlowerAgent._code_binary_set_(inst_gen,CG_routes)
                results['CG_results'].append(((CG_routes,CG_binary_coding,[CG_obj,CG_obj/sum(CG_binary_coding)]),purchase,(CG_missing,CG_missing,CG_extra_cost)))
                results['SL_results'].append((SL_flower_info,SL_purchase,(SL_missing,SL_missing,SL_extra_cost)))
                results['cost_results'].append((cost_flower_info,cost_purchase,(cost_missing,cost_missing,cost_extra_cost)))


                ''' Compound action'''
                action = {'routing':CG_routes,'purchase':purchase,'demand_compliance':demand_compliance}
                state,reward,done,real_action,_, = env.step(action,inst_gen)

                    
                print(f'\tEpisode: {ep_count} - {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
                results['seeds'][1].append((stoch_rd_seed))
                ep_count += 1
                if ep_count == testing_episodes-1: main_done=True
                    
            # except Exception as e:
            #     print(f'❌ \t{env.t} \t{e}')
        
        with open(experiments_path+f'Testing/M{size}-E{episode}.pkl','wb') as file:
                pickle.dump(results,file)
