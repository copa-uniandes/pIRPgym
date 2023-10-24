from numpy.random import seed, random, randint, lognormal


class costs():

    ### Holding cost
    def gen_h_cost(inst_gen,**kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 1)
        if kwargs['dist'] == 'd_uniform':   rd_function = randint
        hist_h = costs.gen_hist_h(inst_gen, rd_function,**kwargs)
        W_h, hist_h = costs.gen_W_h(inst_gen, rd_function, hist_h,**kwargs)

        return hist_h, W_h
    

    # Historic holding cost
    def gen_hist_h(inst_gen,rd_function,**kwargs) -> dict[dict]: 
        hist_h = {t:{} for t in inst_gen.Horizon}
        if inst_gen.other_params['historical'] != False and ('h' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_h[0] = {k:[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products}
        else:
            hist_h[0] = {k:[] for k in inst_gen.Products}

        return hist_h


    # Realized (real) holding cost
    def gen_W_h(inst_gen,rd_function,hist_h,**kwargs) -> tuple:
        '''
        W_h: (dict) holding cost of k \in K  on t \in T
        '''
        W_h = dict()
        for t in inst_gen.Horizon:
            W_h[t] = dict()   
            for k in inst_gen.Products:
                W_h[t][k] = round(rd_function(*kwargs['r_f_params']),2)

                if t < inst_gen.T - 1:
                    hist_h[t+1][k] = hist_h[t][k] + [W_h[t][k]]

        return W_h, hist_h  
