from numpy.random import seed,random,randint,lognormal
from .forecasting import empiric_distribution_sampling

class offer():
    ### Availabilty of products on suppliers
    @staticmethod
    def gen_availabilities(inst_gen) -> tuple:
        '''
        M_kt: (dict) subset of suppliers that offer k in K on t in T
        K_it: (dict) subset of products offered by i in M on t in T
        '''
        seed(inst_gen.d_rd_seed + 3)
        M_kt = dict()
        # In each time period, for each product
        for k in inst_gen.Products:

            sup = randint(3, inst_gen.M+1)
            suppliers = list(inst_gen.Suppliers)
            # Random suppliers are removed from subset, regarding {sup}
            for ss in range(inst_gen.M - sup):
                a = int(randint(0, len(suppliers)))
                del suppliers[a]

            for t in inst_gen.TW:
                M_kt[k,t] = suppliers
        
        # Products offered by each supplier on each time period, based on M_kt
        K_it = {(i,t):[k for k in inst_gen.Products if i in M_kt[k,t]] for i in inst_gen.Suppliers for t in inst_gen.TW}

        return M_kt, K_it
    
    ### Available quantities of products on suppliers
    @staticmethod
    def gen_quantities(inst_gen,**kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 4)
        if kwargs['dist'] == 'c_uniform':   rd_function = randint
        hist_q = offer.gen_hist_q(inst_gen, rd_function, **kwargs)

        if inst_gen.other_params['look_ahead'] != False and ('q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
            seed(inst_gen.s_rd_seed + 1)
            W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)
            s_paths_q = offer.gen_empiric_q_sp(inst_gen, hist_q, W_q)
            return hist_q, W_q, s_paths_q

        else:
            W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)
            return hist_q, W_q, None

    # Historic availabilities
    @staticmethod
    def gen_hist_q(inst_gen,rd_function,**kwargs) -> dict[dict]:
        hist_q = {t:dict() for t in inst_gen.Horizon}
        
        if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_q[0] = {(i,k):[round(rd_function(*kwargs['r_f_params']),2) if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
        else:
            hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

        return hist_q

    
    # Realized (real) availabilities
    @staticmethod
    def gen_W_q(inst_gen,rd_function,hist_q,**kwargs) -> tuple:
        '''
        W_q: (dict) quantity of k in K offered by supplier i in M on t in T
        '''
        W_q = dict()
        for t in inst_gen.Horizon:
            W_q[t] = dict()  
            for i in inst_gen.Suppliers:
                for k in inst_gen.Products:
                    if i in inst_gen.M_kt[k,t]:
                        W_q[t][(i,k)] = round(rd_function(*kwargs['r_f_params']),2)
                    else:   W_q[t][(i,k)] = 0

                    if t < inst_gen.T - 1:
                        hist_q[t+1][i,k] = hist_q[t][i,k] + [W_q[t][i,k]]

        return W_q, hist_q


    # Availabilitie's sample paths
    @staticmethod
    def gen_empiric_q_sp(inst_gen,hist_q,W_q) -> dict[dict]:
        s_paths_q = dict()
        for t in inst_gen.Horizon: 
            s_paths_q[t] = dict()
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('q' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_q[t][0,sample] = W_q[t]
                else:
                    s_paths_q[t][0,sample] = {(i,k): empiric_distribution_sampling([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,0] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_q[t][day,sample] = {(i,k): empiric_distribution_sampling([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

        return s_paths_q


    ### Prices of products on suppliers
    @staticmethod
    def gen_prices(inst_gen, **kwargs) -> tuple:

        seed(inst_gen.d_rd_seed + 5)
        if kwargs['dist'] == 'd_uniform':   rd_function = randint
        hist_p = offer.gen_hist_p(inst_gen, rd_function, **kwargs)
        W_p, hist_p = offer.gen_W_p(inst_gen, rd_function, hist_p, **kwargs)

        if inst_gen.other_params['look_ahead'] != False and ('p' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
            seed(inst_gen.s_rd_seed + 3)
            s_paths_p = offer.gen_empiric_p_sp(inst_gen, hist_p, W_p)
            return hist_p, W_p, s_paths_p 

        else:
            return hist_p, W_p, None
    
    
    # Historic prices
    @staticmethod
    def gen_hist_p(inst_gen, rd_function, **kwargs) -> dict[dict]:
        hist_p = {t:dict() for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('p' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_p[0] = {(i,k):[round(rd_function(*kwargs['r_f_params']),2) if i in inst_gen.M_kt[k,t] else 1000 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
        else:
            hist_p[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

        return hist_p


    # Realized (real) prices
    @staticmethod
    def gen_W_p(inst_gen, rd_function, hist_p, **kwargs) -> tuple:
        '''
        W_p: (dict) quantity of k in K offered by supplier i in M on t in T
        '''
        W_p = dict()
        for t in inst_gen.Horizon:
            W_p[t] = dict()   
            for i in inst_gen.Suppliers:
                for k in inst_gen.Products:
                    if i in inst_gen.M_kt[k,t]:
                        W_p[t][(i,k)] = round(rd_function(*kwargs['r_f_params']),2)
                    else:   W_p[t][(i,k)] = 1000

                    if t < inst_gen.T - 1:
                        hist_p[t+1][i,k] = hist_p[t][i,k] + [W_p[t][i,k]]

        return W_p, hist_p
    

    # Prices's sample paths
    @staticmethod
    def gen_empiric_p_sp(inst_gen, hist_p, W_p) -> dict[dict]:
        
        s_paths_p = dict()
        for t in inst_gen.Horizon: 
            s_paths_p[t] = dict()
            for sample in inst_gen.Samples:
                if inst_gen.s_params == False or ('p' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                    s_paths_p[t][0,sample] = W_p[t]
                else:
                    s_paths_p[t][0,sample] = {(i,k): empiric_distribution_sampling([hist_p[t][i,k][obs] for obs in range(len(hist_p[t][i,k])) if hist_p[t][i,k][obs] < 1000]) if i in inst_gen.M_kt[k,0] else 1000 for i in inst_gen.Suppliers for k in inst_gen.Products}

                for day in range(1,inst_gen.sp_window_sizes[t]):
                    s_paths_p[t][day,sample] = {(i,k): empiric_distribution_sampling([hist_p[t][i,k][obs] for obs in range(len(hist_p[t][i,k])) if hist_p[t][i,k][obs] < 1000]) if i in inst_gen.M_kt[k,t+day] else 1000 for i in inst_gen.Suppliers for k in inst_gen.Products}

        return s_paths_p
    

    class supplier_differentiated():
        ### Available quantities of products on suppliers
        @staticmethod
        def gen_quantities(inst_gen,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 4)
            rd_function = randint
            q_parameters = offer.supplier_differentiated._generate_availability_parameters(inst_gen)
            hist_q = offer.supplier_differentiated._gen_hist_q(inst_gen,rd_function,**q_parameters)

            if inst_gen.other_params['look_ahead'] != False and ('q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed + 1)
                W_q, hist_q = offer.supplier_differentiated._gen_W_q(inst_gen,rd_function,hist_q,**q_parameters)
                s_paths_q = offer.supplier_differentiated._gen_empiric_q_sp(inst_gen,hist_q,W_q)
                return hist_q, W_q, s_paths_q,q_parameters

            else:
                W_q, hist_q = offer.supplier_differentiated._gen_W_q(inst_gen,rd_function,hist_q,**q_parameters)
                return hist_q, W_q, None,q_parameters

        # Historic availabilities
        @staticmethod
        def _gen_hist_q(inst_gen,rd_function,**kwargs) -> dict[dict]:
            hist_q = {t:dict() for t in inst_gen.Horizon}
            
            if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
                hist_q[0] = {(i,k):[round(rd_function(*kwargs[str(i)]),2) if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
            else:
                hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

            return hist_q

        
        # Realized (real) availabilities
        @staticmethod
        def _gen_W_q(inst_gen,rd_function,hist_q,**kwargs) -> tuple:
            '''
            W_q: (dict) quantity of k in K offered by supplier i in M on t in T
            '''
            W_q = dict()
            for t in inst_gen.Horizon:
                W_q[t] = dict()  
                for i in inst_gen.Suppliers:
                    for k in inst_gen.Products:
                        if i in inst_gen.M_kt[k,t]:
                            W_q[t][(i,k)] = round(rd_function(*kwargs[str(i)]),2)
                        else:   W_q[t][(i,k)] = 0

                        if t < inst_gen.T - 1:
                            hist_q[t+1][i,k] = hist_q[t][i,k] + [W_q[t][i,k]]

            return W_q, hist_q


        # Availabilitie's sample paths
        @staticmethod
        def _gen_empiric_q_sp(inst_gen,hist_q,W_q) -> dict[dict]:
            s_paths_q = dict()
            for t in inst_gen.Horizon: 
                s_paths_q[t] = dict()
                for sample in inst_gen.Samples:
                    if inst_gen.s_params == False or ('q' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                        s_paths_q[t][0,sample] = W_q[t]
                    else:
                        s_paths_q[t][0,sample] = {(i,k): empiric_distribution_sampling([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,0] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

                    for day in range(1,inst_gen.sp_window_sizes[t]):
                        s_paths_q[t][day,sample] = {(i,k): empiric_distribution_sampling([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

            return s_paths_q
        
        @staticmethod
        def _generate_availability_parameters(inst_gen):
            for i in inst_gen.Suppliers:
                q_parameters = {str(i):(round(6+11*random()),round(20+13*random())) for i in range(1,inst_gen.M+1)}          # Offer
                for i in range(1,inst_gen.M+1):
                    if q_parameters[str(i)][0] > q_parameters[str(i)][1]:
                        q_parameters[str(i)] = (q_parameters[str(i)][1],q_parameters[str(i)][0])
                    elif q_parameters[str(i)][0] == q_parameters[str(i)][1]:
                        q_parameters[str(i)] = (q_parameters[str(i)][0],q_parameters[str(i)][1]+randint(1,5))
            return q_parameters
