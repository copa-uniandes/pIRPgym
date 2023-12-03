from numpy.random import seed, random, randint, lognormal


class costs():

    ### Holding cost
    @staticmethod
    def gen_h_cost(inst_gen,**kwargs) -> tuple:
        seed(inst_gen.d_rd_seed + 1)
        if kwargs['dist'] == 'd_uniform':   rd_function = randint
        hist_h = costs.gen_hist_h(inst_gen, rd_function,**kwargs)
        W_h, hist_h = costs.gen_W_h(inst_gen, rd_function, hist_h,**kwargs)

        return hist_h, W_h
    

    # Historic holding cost
    @staticmethod
    def gen_hist_h(inst_gen,rd_function,**kwargs) -> dict[dict]: 
        
        hist_h = {t:{} for t in inst_gen.Horizon}

        h_fixed = dict()
        for k in inst_gen.Products:
            historic_avg = {i:sum(inst_gen.hist_p[0][i,k][t+inst_gen.hist_window] for t in inst_gen.historical)/inst_gen.hist_window for i in inst_gen.M_kt[k,0]}
            h_fixed[k] = sum(historic_avg.values())/(len(historic_avg)*(inst_gen.O_k[k]+1))
        
        if inst_gen.other_params['historical'] != False and ('h' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
            hist_h[0] = {k:[h_fixed[k] for t in inst_gen.historical] for k in inst_gen.Products}
        else:
            hist_h[0] = {k:[] for k in inst_gen.Products}

        return hist_h


    # Realized (real) holding cost
    @staticmethod
    def gen_W_h(inst_gen,rd_function,hist_h,**kwargs) -> tuple:
        '''
        W_h: (dict) holding cost of k in K  on t in T
        '''
        W_h = dict()
        for t in inst_gen.Horizon:
            W_h[t] = dict()   
            for k in inst_gen.Products:
                W_h[t][k] = hist_h[0][k][0]

                if t < inst_gen.T - 1:
                    hist_h[t+1][k] = hist_h[t][k] + [W_h[t][k]]

        return W_h, hist_h 
    
    @staticmethod
    def gen_profit_margin(inst_gen, **kwargs):

        profit = dict()
        seed(inst_gen.d_rd_seed + 10)
        for k in inst_gen.Products:
            profit[k] = 0.1+0.65*random()
        
        return profit

    @staticmethod
    def gen_backo_cost(inst_gen,**kwargs):

        back_o_cost = dict()
        for k in inst_gen.Products:
            historic_avg = {i:sum(inst_gen.hist_p[0][i,k][t+inst_gen.hist_window] for t in inst_gen.historical)/inst_gen.hist_window for i in inst_gen.M_kt[k,0]}
            back_o_cost[k] = (inst_gen.prof_margin[k])*sum(historic_avg.values())/len(historic_avg)
        
        return back_o_cost
            


