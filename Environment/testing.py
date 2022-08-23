from SD_IB_IRP_PPenv import steroid_IRP

### Parameters
look_ahead = ['*']

S = 4
LA_horizon = 3

historical_data = ['*']

hist_window = 10

backorders =  'backorders'

env_config = {  'M': 10, 
                'K': 10, 
                'T': 7, 
                'F': 4, 
                
                
                'S': S, 
                'LA_horizon': LA_horizon, 
            }
            

rd_seed = 0
env = steroid_IRP( look_ahead = look_ahead, 
                   historical_data = historical_data, 
                   backorders = backorders,
                   rd_seed = rd_seed, 
                   env_config = env_config)

