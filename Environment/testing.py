from SD_IB_IRP_PPenv import steroid_IRP
### Instance generation
from InstanceGenerator import instance_generator

### Parameters
rd_seed = 0
look_ahead = ['q','d']
S = 4
LA_horizon = 3
historical_data = ['*']
hist_window = 10000
backorders =  'backorders'
stochastic_parameters = ['q','d']
env_config = {  'M': 10, 
                'K': 10, 
                'T': 7, 
                'F': 4, 
                
                'S': S, 
                'LA_horizon': LA_horizon, 
            }         


env = steroid_IRP( look_ahead = look_ahead, 
                   historical_data = historical_data, 
                   backorders = backorders,
                   stochastic_parameters = stochastic_parameters, 
                   env_config = env_config)

q_params = {'distribution': 'c_uniform', 'min': 100, 'max': 200}
d_params = {'distribution': 'log-normal', 'mean': 500, 'stdev': 20}

p_params = {'distribution': 'd_uniform', 'min': 20, 'max': 60}
h_params = {'distribution': 'd_uniform', 'min': 20, 'max': 60}

env.reset(return_state = True, rd_seed = rd_seed, 
          q_params = q_params, 
          p_params = p_params,
          d_params = d_params,
          h_params = h_params)


