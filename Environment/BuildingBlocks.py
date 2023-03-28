"""
@author: juanbeta
Independent transition blocks of a Supply Chain
"""
from numpy.random import seed, random, randint, lognormal


class routing():

    def __init__(self):
        pass
    

    def evaluate_solution(self, instance, routes):
        transport_cost = 0
        for route in routes:
            transport_cost += sum(instance.c[route[i], route[i + 1]] for i in range(len(route) - 1))
        return transport_cost


class Inventory_management():
    
    class perishable_per_age_inv():

        def reset(self, inst_gen):
            ## State ##
            seed(inst_gen.d_rd_seed*2)
            max_age = inst_gen.T
            O_k = {k:randint(1,max_age+1) for k in inst_gen.Products} 
            
            state = {(k,o):0   for k in inst_gen.Products for o in range(1, self.O_k[k] + 1)}
            if inst_gen.other_params['backorders'] == 'backlogs':
                for k in inst_gen.Products:
                    self.state[k,'B'] = 0
            
            return state, O_k


