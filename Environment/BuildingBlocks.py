
"""
@author: juanbeta

### Independent transition blocks of a Supply Chain

"""

class routing():

    def __init__(self):
        pass
    

    def evaluate_solution(self, instance, routes):
        transport_cost = 0
        for route in routes:
            transport_cost += sum(instance.c[route[i], route[i + 1]] for i in range(len(route) - 1))
        return transport_cost


class perishable_per_age_inv():

    def __init__():
        pass


    def reset(self, M, O_k):
        ## State ##
        self.state = {(k,o):0   for k in self.Products for o in range(1, self.O_k[k] + 1)}
        if self.other_env_params['backorders'] == 'backlogs':
            for k in self.Products:
                self.state[k,'B'] = 0


