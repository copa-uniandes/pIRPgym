import numpy as np
from .Policies.Inventory import Inventory

class Compromise_Programming():

    @staticmethod
    def normalize_objectives(inst_gen, env):
        
        env.payoff_matrix["costs"] = Inventory.Stochastic_Rolling_Horizon(env.state,env,inst_gen)
        for e in inst_gen.E:
            env.payoff_matrix[e] = Inventory.Stochastic_Rolling_Horizon(env.state,env,inst_gen,objs={e:1})
        
        env.norm_matrix["costs"]["best"] = env.payoff_matrix["costs"]["costs"]
        env.norm_matrix["costs"]["worst"] = np.max([env.payoff_matrix[e]["costs"] for e in inst_gen.E])
        for e in inst_gen.E:
            env.norm_matrix[e]["best"] = env.payoff_matrix[e][e]
            env.norm_matrix[e]["worst"] = np.max([env.payoff_matrix[ee][e] for ee in inst_gen.E+["costs"]])
    
    @staticmethod
    def normalize_objectives_IRP(inst_gen, env, verbose=False):

        ''' Get payoff matrix '''
        m, variables, costs, impacts = Inventory.IRP.build_model(env.state, env, inst_gen)
        #env.payoff_matrix["costs"] = Inventory.Stochastic_RH_IRP(env.state,env,inst_gen, verbose=verbose)
        for e in inst_gen.E:
            env.payoff_matrix[e] = Inventory.optimize_environmental_indicator(e, m, impacts, env.state, env, inst_gen, verbose = verbose)
        
        ''' Get normalization bounds '''
        for e in inst_gen.E+["costs"]:
            env.norm_matrix[e]["best"] = env.payoff_matrix[e][e]
            env.norm_matrix[e]["worst"] = np.max([env.payoff_matrix[ee][e] for ee in inst_gen.E+["costs"]])

        