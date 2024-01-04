#%%##########################################       Modules       ###########################################
# MODULES
import sys
from time import process_time
import numpy as np;from numpy import random
import pickle
import datetime

sys.path.append('../../../.')
import pIRPgym

computer_name = input("Running experiment on mac? [Y/n]")
if computer_name == '': 
    path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/pIRPgym/'
    experiments_path = '/Users/juanbeta/My Drive/Research/Supply Chain Analytics/Experiments/Flower Agent/'
else: 
    path = 'C:/Users/jm.betancourt/Documents/Research/pIRPgym/'
    experiments_path = 'G:/Mi unidad/Research/Supply Chain Analytics/Experiments/Flower Agent/'

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

testing_episodes = 10

Num_Episodes = [10,30,50]
Num_Suppliers = {10:[10,20,30,40,50],
                  30:[5,10,15,20,30],
                  50:[20,30]}

identifiers = {(10,10):0,(10,30):1,(10,50):2,(30,10):3,(30,30):4,(30,50):5,(50,10):6,(50,30):7,(50,50):8}


for episode in Num_Episodes[::]:
    for size in Num_Suppliers[episode][::]:
        print(f'M{size}-E{episode}')
        main_done = False
        ep_count = 0
        
        with open(experiments_path+f'Training/M{size}-E{episode}.pkl', 'rb') as file:
            _,_,hist_inst_gen,FlowerAgent = pickle.load(file)

        # Creating instance generator object
        env_config['M']=size
        env_config['K']=env_config['M']
        env_config['F']=env_config['M']
        inst_gen = pIRPgym.instance_generator(look_ahead,stochastic_params,historical_data,backorders,
                                              env_config=env_config)
        
        results = {'info':(inst_gen,episode,size),'TP_results':list(),
                   'SL_results':list(),'C_results':list()}

        det_rd_seed = hist_inst_gen.d_rd_seed             # Random seeds
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
                ''' Two Phases Action '''
                # Purchasing
                [purchase,demand_compliance],la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)
                total_purchase = sum(purchase.values())
                purchase_cost = purchase_cost = {k:sum(purchase[i,k] * inst_gen.W_p[env.t][i,k]   for i in inst_gen.Suppliers) for k in inst_gen.Products}
                price_delta = pIRPgym.Routing_management.evaluate_purchase(inst_gen,purchase,env.t)
                # Routing
                TwPh_routes,TwPh_obj,TwPh_info,TwPh_time,TwPh_cols = pIRPgym.Routing.ColumnGeneration(purchase,inst_gen,env.t,time_limit=1,
                                                                                            verbose=False,heuristic_initialization=0.2,
                                                                                            return_num_cols=True,RCL_alpha=0.8) 

                ''' Fitting routing to purchase '''
                SL_flower_info,C_flower_info,solution_time = FlowerAgent.fit_purchase_to_flower(purchase,inst_gen,env.t,n=5)
                # Service Level
                [SL_purchase,SL_demand_compliance],SL_la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen,fixed_suppliers=SL_flower_info[1])
                SL_purchase_cost = {k:sum(SL_purchase[i,k] * inst_gen.W_p[env.t][i,k]   for i in inst_gen.Suppliers) for k in inst_gen.Products}
                total_SL_purchase = sum(SL_purchase.values())

                # Cost
                [C_purchase,C_demand_compliance],C_la_dec = pIRPgym.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen,fixed_suppliers=C_flower_info[1])
                C_purchase_cost = {k:sum(C_purchase[i,k] * inst_gen.W_p[env.t][i,k]   for i in inst_gen.Suppliers) for k in inst_gen.Products}
                total_C_purchase = sum(C_purchase.values())


                ''' Evaluating policies '''
                TwPh_missing,TwPh_r_missing,TwPh_extra_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,TwPh_routes,purchase,False)
                SL_missing,SL_r_missing,SL_extra_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,SL_flower_info[0],purchase,False)
                C_missing,C_r_missing,C_extra_cost = pIRPgym.Routing_management.evaluate_solution_dynamic_potential(inst_gen,env,C_flower_info[0],purchase,False)


                ''' Saving results'''
                # Two phases
                TwPh_binary_coding = FlowerAgent._code_binary_set_(inst_gen,TwPh_routes)
                TwPh_metrics = [TwPh_obj,TwPh_obj/sum(TwPh_binary_coding)]
                TwPh_flower_info = (TwPh_routes,TwPh_binary_coding,TwPh_metrics)
                TwPh_purchase_info = (purchase,purchase_cost)
                TwPh_service_levels = (1-TwPh_missing/total_purchase,1-TwPh_r_missing/total_purchase)
                results['TP_results'].append((TwPh_flower_info,TwPh_purchase_info,TwPh_service_levels))

                # Service Level
                SL_purchase_info = (SL_purchase,SL_purchase_cost)
                SL_service_levels = (1-SL_missing/total_SL_purchase,1-SL_r_missing/total_SL_purchase)
                results['SL_results'].append((SL_flower_info,SL_purchase_info,SL_service_levels))

                # Cost
                C_purchase_info = (C_purchase,C_purchase_cost)
                C_service_levels = (1-C_missing/total_C_purchase,1-C_r_missing/total_C_purchase)
                results['C_results'].append((C_flower_info,C_purchase_info,C_service_levels))


                ''' Compound action'''
                action = {'routing':TwPh_routes,'purchase':purchase,'demand_compliance':demand_compliance}
                state,reward,done,real_action,_, = env.step(action,inst_gen)

                    
                print(f'\tEpisode: {ep_count} - {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
                results['seeds'][1].append((stoch_rd_seed))
                ep_count += 1
                if ep_count == testing_episodes-1: main_done=True
                    
            # except Exception as e:
            #     print(f'❌ \t{env.t} \t{e}')
        
        with open(experiments_path+f'Testing/M{size}-E{episode}.pkl','wb') as file:
                pickle.dump(results,file)
