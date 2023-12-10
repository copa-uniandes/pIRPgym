from ..InstanceGenerator import instance_generator

class Purchasing():
    options = ['det_purchase_all', 'avg_purchase_all']
    
    # Purchases all available quantities assuming deterministic available quantities of each supplier
    @staticmethod
    def det_purchase_all(inst_gen:instance_generator, env) -> dict[tuple:float]:
        purchase = dict()
        for i in inst_gen.Suppliers:
            for k in inst_gen.Products:
                purchase[(i,k)] = inst_gen.W_q[env.t][i,k]

        return purchase
    

    # Purchases expected value of available quantities of each supplier
    @staticmethod
    def avg_purchase_all(inst_gen:instance_generator,env)->dict:
        purchase = dict()
        for i in inst_gen.Suppliers:
            for k in inst_gen.Products:
                purchase[(i,k)] = sum(inst_gen.s_paths_q[env.t][0,s][i,k] for s in inst_gen.Samples)/inst_gen.S
        
        return purchase
