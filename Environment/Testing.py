# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 09:24:41 2022

@author: juanbeta
"""
from SD_IB_IRP_PPenv import steroid_IRP
from random import choice, randint
from termcolor import colored

rd_seed = 0

### Environment initalization parameters ###
horizon_type = 'episodic'
T = 6

look_ahead = ['*']
S = 5
LA_horizon = 4

historic_data = ['*']
hist_window = 10

env_config = {'M': 10, 'K': 10, 'T': T, 'F': 2, 'min_sprice': 1, 'max_sprice': 500, 'min_hprice': 1, 
              'max_hprice': 500, 'S': S, 'LA_horizon': LA_horizon, 'lambda1': 0.5 }

env = steroid_IRP(  horizon_type = horizon_type,
                    look_ahead = look_ahead, 
                    historic_data = historic_data, 
                    rd_seed = rd_seed, 
                    env_config = env_config)

return_state = False
env.reset(return_state = return_state)

# Visiting all the suppliers
routes = [[0,1,2,0]]

# Purchase exact quantity for 
purchase = {(1,0): 6, (2,0): 4, 
            (1,1): 8, (2,1): 10,
            (1,2): 0, (2,2): 0}

demand_complience = {(k,o):0 for k in env.Products for o in range(env.O_k[k])}

X = [routes, purchase, demand_complience]

state, reward, done, _ = env.step(action = X)

print(state)
