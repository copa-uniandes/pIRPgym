from numpy.random import seed, random, randint, lognormal
# from .InstanceGenerator import instance_generator


class demand():
    ### Demand of products
    def gen_demand(inst_gen, **kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 2)
        if kwargs['dist'] == 'log-normal':   rd_function = lognormal
        elif kwargs['dist'] == 'd_uniform': rd_function = randint

        hist_d = demand.gen_hist_d(inst_gen, rd_function, **kwargs)

        if inst_gen.other_params['look_ahead'] != False and ('d' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
            seed(inst_gen.s_rd_seed)
            W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)
            s_paths_d = demand.gen_empiric_d_sp(inst_gen, hist_d, W_d)
            return hist_d, W_d, s_paths_d

        else:
            W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)
            return hist_d, W_d, None
    
    ### Demand of products
    def gen_demand_age(inst_gen, **kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 2)
        if kwargs['dist'] == 'log-normal':   rd_function = lognormal
        elif kwargs['dist'] == 'd_uniform': rd_function = randint

        hist_d = demand.gen_hist_d_age(inst_gen, rd_function, **kwargs)

        if 'd' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']:
            seed(inst_gen.s_rd_seed)
            W_d, hist_d = demand.gen_W_d_age(inst_gen, rd_function, hist_d, **kwargs)
            s_paths_d = demand.gen_empiric_d_sp_age(inst_gen, hist_d, W_d)
            return hist_d, W_d, s_paths_d

        else:
            W_d, hist_d = demand.gen_W_d_age(inst_gen, rd_function, hist_d, **kwargs)
            return hist_d, W_d, None

    # Historic demand
    def gen_hist_d(inst_gen, rd_function, **kwargs) -> dict[dict]: 
        hist_d = {t:dict() for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('d' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_d[0] = {k:[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products}
        else:
            hist_d[0] = {k:[] for k in inst_gen.Products}

        return hist_d

    # Historic demand for age-dependent demand
    def gen_hist_d_age(inst_gen, rd_function, **kwargs) -> dict[dict]: 
        hist_d = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('d' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            r_f_params = kwargs.get("r_f_params")
            #hist_d[0] = {(k,o):[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}
            hist_d[0] = {(k,o):[round(rd_function(r_f_params[o][0],r_f_params[o][1]),2) for t in inst_gen.historical] for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}
        else:
            hist_d[0] = {(k,o):[] for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}

        return hist_d


    # Realized (real) availabilities
    def gen_W_d(inst_gen, rd_function, hist_d, **kwargs) -> tuple:
        '''
        W_d: (dict) demand of k \in K  on t \in T
        '''
        W_d = dict()
        for t in inst_gen.Horizon:
            W_d[t] = dict()   
            for k in inst_gen.Products:
                W_d[t][k] = round(rd_function(*kwargs['r_f_params']),2)

                if t < inst_gen.T - 1:
                    hist_d[t+1][k] = hist_d[t][k] + [W_d[t][k]]

        return W_d, hist_d
    
    # Realized (real) availabilities
    def gen_W_d_age(inst_gen, rd_function, hist_d, **kwargs) -> tuple:
        '''
        W_d: (dict) demand of k \in K  on t \in T
        '''
        r_f_params = kwargs.get("r_f_params")
        W_d = {}
        for t in inst_gen.Horizon:
            W_d[t] = {}   
            for k in inst_gen.Products:
                for o in range(inst_gen.O_k[k]+1):
                    W_d[t][k,o] = round(rd_function(r_f_params[o][0],r_f_params[o][1]),2)

                    if t < inst_gen.T - 1:
                        hist_d[t+1][k,o] = hist_d[t][k,o] + [W_d[t][k,o]]

        return W_d, hist_d

    # Demand's sample paths
    def gen_empiric_d_sp(inst_gen, hist_d, W_d) -> dict[dict]:
        s_paths_d = dict()
        for t in inst_gen.Horizon: 
            s_paths_d[t] = dict()
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('d' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_d[t][0,sample] = W_d[t]
                else:
                    s_paths_d[t][0,sample] = {k: inst_gen.sim([hist_d[t][k][obs] for obs in range(len(hist_d[t][k])) if hist_d[t][k][obs] > 0]) for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_d[t][day,sample] = {k: inst_gen.sim([hist_d[t][k][obs] for obs in range(len(hist_d[t][k])) if hist_d[t][k][obs] > 0]) for k in inst_gen.Products}

        return s_paths_d
    
    # Demand's sample paths
    def gen_empiric_d_sp_age(inst_gen, hist_d, W_d) -> dict[dict]:
        s_paths_d = {}
        for t in inst_gen.Horizon: 
            s_paths_d[t] = {}
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('d' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_d[t][0,sample] = W_d[t]
                else:
                    s_paths_d[t][0,sample] = {(k,o): inst_gen.sim([hist_d[t][k,o][obs] for obs in range(len(hist_d[t][k,o])) if hist_d[t][k,o][obs] > 0]) for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_d[t][day,sample] = {(k,o): inst_gen.sim([hist_d[t][k,o][obs] for obs in range(len(hist_d[t][k,o])) if hist_d[t][k,o][obs] > 0]) for k in inst_gen.Products for o in range(inst_gen.O_k[k]+1)}

        return s_paths_d
        