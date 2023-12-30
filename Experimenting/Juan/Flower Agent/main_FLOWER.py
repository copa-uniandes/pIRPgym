#%%##########################################       Modules       ###########################################
# MODULES
import sys
from time import process_time
import numpy as np;from numpy import random
import pickle

sys.path.append('../../../.')
import pIRPgym

computer_name = input("Running experiment on mac? [Y/n]")
if computer_name == '': 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Flower Agent/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Flower Agent/'


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

sizes = [5,10,15,20,30]
env_config = {'T':12,'Q':750,
              'S':3,'LA_horizon':3,
             'd_max':2500,'hist_window':60,
             'theta':0.7}




##########################################    Random Instance    ##########################################
# Random Instance
# q_params = {'dist':'c_uniform','r_f_params':(6,20)}          # Offer
p_params = {'dist':'d_uniform','r_f_params':(20,61)}

d_params = {'dist':'log-normal','r_f_params':(3,1)}          # Demand

h_params = {'dist':'d_uniform','r_f_params':(20,61)}         # Holding costs
     


disc = ("strong","conc")


#%% 
# Environment
# Creating environment object
env = pIRPgym.steroid_IRP(True,True,True)

seeds = []

FlowerAgent = pIRPgym.FlowerAgent(solution_num=50)
main_done = False
ep_count = 0
num_episodes = 100




env_config['M']=sizes[0]
env_config['K']=env_config['M']
env_config['F']=env_config['M']
det_rd_seed = env_config['K']             # Random seeds

# Creating instance generator object
inst_gen = pIRPgym.instance_generator(look_ahead,stochastic_params,
                        historical_data,backorders,env_config=env_config)


stoch_rd_seed = det_rd_seed*10000  

print(f'Suppliers: {env_config["M"]} - Episodes: {num_episodes}')
while not main_done:
    stoch_rd_seed+=1
    try:
        inst_gen.generate_supplier_differentiated_random_instance(det_rd_seed,stoch_rd_seed,p_params=p_params,
                                                                    d_params=d_params,h_params=h_params,discount=disc)
        state = env.reset(inst_gen,return_state=True)
        done = False
        while not done:
            ''' Purchase '''
            [purchase,demand_compliance],la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
            total_purchase = sum(purchase.values())
            price_delta = pIRPgym.Routing_management.evaluate_purchase(inst_gen,purchase,env.t)


            ''' Generating solutions '''
            # Genetic Algorithm
            GA_routes,GA_obj,GA_info,GA_time,_ = pIRPgym.Routing.GeneticAlgorithm(purchase,inst_gen,env.t,return_top=False,
                                                                                rd_seed=0,time_limit=150,verbose=False)    # Genetic Algorithm
            # Column Generations
            CG_routes,CG_obj,CG_info,CG_time,CG_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=300,
                                                                                    verbose=False,heuristic_initialization=1,
                                                                                    return_num_cols=True,RCL_alpha=0.6) 

            ''' Update flower pool '''
            # Genetic Algorithm
            GA_tot_mis,GA_rea_mis,GA_e_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,GA_routes,purchase,
                                                                                                            discriminate_missing=False)
            GA_SL = 1-GA_tot_mis/total_purchase; GA_reactive_SL = 1-GA_rea_mis/total_purchase
            FlowerAgent.update_flower_pool(inst_gen,GA_routes,generator='GA',cost=GA_obj,total_SL=GA_SL,reactive_SL=GA_reactive_SL,price_delta=price_delta)

            # Column Generation
            CG_tot_mis,CG_rea_mis,CG_e_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,CG_routes,purchase,
                                                                                                            discriminate_missing=False)
            CG_SL = 1-CG_tot_mis/total_purchase; CG_reactive_SL = 1-CG_rea_mis/total_purchase
            FlowerAgent.update_flower_pool(inst_gen,CG_routes,generator='CG',cost=CG_obj,total_SL=CG_SL,reactive_SL=CG_reactive_SL,price_delta=price_delta)


            ''' Compound action'''
            action = {'routing':CG_routes,'purchase':purchase,'demand_compliance':demand_compliance}
            state,reward,done,real_action,_,  = env.step(action,inst_gen)

            
        print(f'Episode: {ep_count}')
        seeds.append((stoch_rd_seed))
        ep_count += 1
        if ep_count == num_episodes: main_done=True
        
    except:
        print('❌')


with open(experiments_path+f'M{env_config["M"]}-{num_episodes}.pkl','wb') as file:
        pickle.dump([env_config['M'],seeds,inst_gen,FlowerAgent],file)


#%%


