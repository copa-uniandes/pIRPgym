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
T = 4

look_ahead = ['*']
S = 2
LA_horizon = 3

historic_data = ['*']
hist_window = 10

env_config = {'M': 3, 'K': 3, 'T': T, 'F': 2, 'min_sprice': 1, 'max_sprice': 500, 'min_hprice': 1, 
              'max_hprice': 500, 'S': S, 'LA_horizon': LA_horizon, 'lambda1': 0.5 }

env = steroid_IRP(horizon_type = horizon_type, look_ahead = look_ahead, historic_data = historic_data, rd_seed = rd_seed, env_config = env_config)

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


















""" print(env.state, '\n')

product = choice(env.Products); age = randint(0, env.O_k[product] - 1)
print(env.O_k[product])
print(f'The inventory of product {product} and age {age} is {env.state[product,age]}')

#print(env.q, '\n')
supplier = choice(env.Suppliers); product = choice(env.Products)
print(f'Supplier {supplier} offers {env.q[supplier, product]} of product {product}')

#print(env.p, '\n')
print(f'Supplier {supplier} offers product {product} at ${env.p[supplier, product]}')

#print(env.h, '\n')
product = choice(env.Products)
print(f'Holding cost for product {product} is {env.h[product]}')

#print(env.h, '\n')
product = choice(env.Products)
print(f'Demand for product {product} is {env.d[product]}')


supplier = choice(env.Suppliers); product = choice(env.Products)

historic_quantities = env.historic_data['q'][supplier, product]
print(f'The historic a.q. for supplier {supplier} on product {product} are {historic_quantities} \n')

historic_demand = env.historic_data['d'][product]
print(f'The historic demand of produdct {product} is {historic_demand}')

sample = choice(env.Samples)
proy_day = randint(1, env.LA_horizon - 1)
supplier = choice(env.Suppliers); product = choice(env.Products)

proy_quant = env.sample_paths[('q',sample)][(supplier, product, proy_day)]
print(f'The forcasted available quantity of product {product} on supplier {supplier} for day {proy_day} is {proy_quant}')

proy_demand = env.sample_paths[('d',sample)][product, proy_day]
print(f'The forcasted demand of product {product} for day {proy_day} is {proy_demand}')

sample = choice(env.Samples)
proy_day = 0
supplier = choice(env.Suppliers); product = choice(env.Products)

def print_valid(statement):  
    if statement:  return 'Passed', 'green'
    else: return 'Failed', 'red'

test_q, col_q = print_valid(env.sample_paths["q",sample][supplier,product,proy_day] == env.q[supplier,product])
print('Test q:', colored(test_q, col_q))
test_p, col_p = print_valid(env.sample_paths["p",sample][supplier,product,proy_day] == env.p[supplier,product])
print('Test p:', colored(test_p, col_p))
test_h, col_h = print_valid(env.sample_paths["h",sample][product,proy_day] == env.h[product])
print('Test h:', colored(test_h, col_h))
test_d, col_d = print_valid(env.sample_paths["d",sample][product,proy_day] == env.d[product])
print('Test h:', colored(test_d, col_d)) """


