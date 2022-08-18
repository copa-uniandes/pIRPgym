import InstanceGenerator as Gen


instance_config = { 'M': 20, 
                    'K': 20, 
                    'T': 7, 
                    'F': 4}
    

instance = Gen.instance_generator(env_config = instance_config)

par = {}
par['q'] = {}
par['q']['distribution'] = 'normal'
par['q']['mean'] = 20
par['q']['stdev'] = 50

par['d'] = {}
par['d']['distribution'] = 'log-normal'
par['d']['mean'] = 20
par['d']['stdev'] = 50

par['p'] = {}
par['p']['distribution'] = 'normal'
par['p']['mean'] = 1
par['p']['stdev'] = 500

par['h'] = {}
par['h']['distribution'] = 'normal'
par['h']['mean'] = 1
par['h']['stdev'] = 500

instance.gen_availabilities()
