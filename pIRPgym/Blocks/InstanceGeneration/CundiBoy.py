import os
import pandas as pd
import numpy as np
from numpy.random import seed, randint,lognormal

class CundiBoy():
    
    def upload_instance():
        data_suppliers = pd.read_excel(os.getcwd()+'/Data/Data_Fruver_0507.xlsx', sheet_name='provider_orders')
        data_demand = pd.read_excel(os.getcwd()+"/Data/Data_Fruver_0507.xlsx",sheet_name="daily_sales_historic")
        data_demand = data_demand[["date","store_product_id","sales"]]

        K = list(pd.unique(data_demand["store_product_id"]))

        data_demand = data_demand.groupby(by=["date","store_product_id"],as_index=False).sum()
        hist_demand = {k:[data_demand.loc[i,"sales"] for i in data_demand.index if data_demand.loc[i,"store_product_id"] == k and data_demand.loc[i,"sales"] > 0] for k in K}
        total_sales = pd.DataFrame.from_dict({k:sum(hist_demand[k]) for k in K},orient="index")
        df_sorted = total_sales.sort_values(by=0, ascending=False)

        # Retrieve the top 30 products with the highest sales
        K = list(df_sorted.head(30).index)
        hist_demand = {k:hist_demand[k] for k in K}


        M = list()
        M_names = list()

        M_k = dict()
        K_i = dict()

        ordered = dict()
        delivered = dict()

        prices = dict()

        for obs in data_suppliers.index:
            i = data_suppliers['provider_id'][obs]
            k = data_suppliers['store_product_id'][obs]

            if k in K:
                if i not in M:
                    M.append(i)
                    M_names.append(data_suppliers["provider_name"])
                    K_i[i] = list()
                
                if k not in M_k.keys():
                    M_k[k] = list()

                M_k[k].append(i)
                K_i[i].append(k)

                if (i,k) not in ordered.keys():
                    ordered[i,k] = 0
                    delivered[i,k] = 0
                    prices[i,k] = list()
                
                ordered[i,k] += data_suppliers['quantity_order'][obs]
                delivered[i,k] += data_suppliers['quantity_received'][obs]

                prices[i,k].append(data_suppliers['cost'][obs])
            

        for i in M:
            K_i[i] = set(K_i[i])
            K_i[i] = list(K_i[i])

        for k in K:
            M_k[k] = set(M_k[k])
            M_k[k] = list(M_k[k])


        service_level = dict()
        for (i,k) in ordered.keys():
            service_level[i,k] = delivered[i,k]/ordered[i,k]

        ex_q = dict()
        ex_d = dict()
        ex_p = dict()
        for k in K:
            ex_d[k] = sum(hist_demand[k])/321

            target_demand = ex_d[k]*1.5
            total_vals = sum([service_level[i,k] for i in M_k[k]])

            for i in M:
                if k in K_i[i]:
                    ex_q[i,k] = target_demand*service_level[i,k]/total_vals
                    ex_p[i,k] = sum(prices[i,k])/len(prices[i,k])
                else:
                    ex_q[i,k] = 0

        return M, M_names, K, M_k, K_i, ex_q, ex_d, ex_p, service_level, hist_demand
    

    class demand():
        ### Demand of products
        def gen_demand(inst_gen,ex_d,hist_demand,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 2)
            if kwargs['dist'] == 'log-normal':   rd_function = lognormal
            elif kwargs['dist'] == 'd_uniform': rd_function = randint

            hist_d = {t:dict() for t in inst_gen.Horizon}
            hist_d.update({0:{k:hist_demand[k] for k in inst_gen.Products}})
            

            if inst_gen.other_params['look_ahead'] != False and ('d' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed)
                W_d, hist_d = CundiBoy.demand.gen_W_d(inst_gen,ex_d,rd_function,hist_d,**kwargs)
                s_paths_d = CundiBoy.demand.gen_empiric_d_sp(inst_gen, hist_d, W_d)
                return hist_d, W_d, s_paths_d

            else:
                W_d, hist_d = demand.gen_W_d(inst_gen, rd_function, hist_d, **kwargs)
                return hist_d, W_d, None

        # Historic demand
        def gen_hist_d(inst_gen, rd_function, **kwargs) -> dict[dict]: 
            hist_d = {t:dict() for t in inst_gen.Horizon}
            if inst_gen.other_params['historical'] != False and ('d' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
                hist_d[0] = {k:[round(rd_function(*kwargs['r_f_params']),2) for t in inst_gen.historical] for k in inst_gen.Products}
            else:
                hist_d[0] = {k:[] for k in inst_gen.Products}

            return hist_d


        # Realized (real) availabilities
        def gen_W_d(inst_gen,ex_d,rd_function,hist_d,**kwargs) -> tuple:
            '''
            W_d: (dict) demand of k \in K  on t \in T
            '''
            W_d = dict()
            for t in inst_gen.Horizon:
                W_d[t] = dict()
                for k in inst_gen.Products:
                    mean_parameter = np.log(ex_d[k]) - 0.5 * np.log(1 + kwargs['r_f_params'] / ex_d[k]**2)
                    sigma = np.sqrt(np.log(1 + kwargs['r_f_params'] / ex_d[k]**2))
                    W_d[t][k] = lognormal(mean_parameter,sigma)

                    if t < inst_gen.T - 1:
                        hist_d[t+1][k] = hist_d[t][k] +[W_d[t][k]]

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

        
    class offer():
        ### Available quantities of products on suppliers
        def gen_quantities(inst_gen,ex_q,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 4)
            if kwargs['dist'] == 'c_uniform':   rd_function = randint
            hist_q = CundiBoy.offer.gen_hist_q(inst_gen,ex_q,rd_function,**kwargs)

            if inst_gen.other_params['look_ahead'] != False and ('q' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed + 1)
                W_q, hist_q = CundiBoy.offer.gen_W_q(inst_gen, ex_q, rd_function, hist_q, **kwargs)
                s_paths_q = CundiBoy.offer.gen_empiric_q_sp(inst_gen, hist_q, W_q)
                return hist_q, W_q, s_paths_q 

            else:
                W_q, hist_q = offer.gen_W_q(inst_gen, rd_function, hist_q, **kwargs)
                return hist_q, W_q, None


        # Historic availabilities
        def gen_hist_q(inst_gen,ex_q,rd_function,**kwargs) -> dict[dict]:
            hist_q = {t:dict() for t in inst_gen.Horizon}
            if inst_gen.other_params['historical'] != False and ('q' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
                hist_q[0] = {(i,k):[max(round(rd_function(ex_q[i,k]-kwargs['r_f_params'],ex_q[i,k]+kwargs['r_f_params']),2),0) if i in inst_gen.M_kt[k,t] else 0 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
            else:
                hist_q[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

            return hist_q

        
        # Realized (real) availabilities
        def gen_W_q(inst_gen,ex_q,rd_function,hist_q,**kwargs) -> tuple:
            '''
            W_q: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            '''
            W_q = dict()
            for t in inst_gen.Horizon:
                W_q[t] = dict()  
                for i in inst_gen.Suppliers:
                    for k in inst_gen.Products:
                        if i in inst_gen.M_kt[k,t]:
                            W_q[t][(i,k)] = max(round(rd_function(ex_q[i,k]-kwargs['r_f_params'],ex_q[i,k]+kwargs['r_f_params']),2),0)
                        else:   W_q[t][(i,k)] = 0

                        if t < inst_gen.T - 1:
                            hist_q[t+1][i,k] = hist_q[t][i,k] + [W_q[t][i,k]]

            return W_q, hist_q


        # Availabilitie's sample paths
        def gen_empiric_q_sp(inst_gen, hist_q, W_q) -> dict[dict]:
            s_paths_q = dict()
            for t in inst_gen.Horizon: 
                s_paths_q[t] = dict()
                for sample in inst_gen.Samples:
                    if inst_gen.s_params == False or ('q' not in inst_gen.s_params and '*' not in inst_gen.s_params):
                        s_paths_q[t][0,sample] = W_q[t]
                    else:
                        s_paths_q[t][0,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,0] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

                    for day in range(1,inst_gen.sp_window_sizes[t]):
                        s_paths_q[t][day,sample] = {(i,k): inst_gen.sim([hist_q[t][i,k][obs] for obs in range(len(hist_q[t][i,k])) if hist_q[t][i,k][obs] > 0]) if i in inst_gen.M_kt[k,t+day] else 0 for i in inst_gen.Suppliers for k in inst_gen.Products}

            return s_paths_q


        ### Prices of products on suppliers
        def gen_prices(inst_gen,ex_p,**kwargs) -> tuple:
            seed(inst_gen.d_rd_seed + 5)
            if kwargs['dist'] == 'd_uniform':   rd_function = randint
            hist_p = CundiBoy.offer.gen_hist_p(inst_gen,rd_function,ex_p,**kwargs)
            W_p, hist_p = CundiBoy.offer.gen_W_p(inst_gen,rd_function, hist_p,ex_p,**kwargs)

            if inst_gen.other_params['look_ahead'] != False and ('p' in inst_gen.other_params['look_ahead'] or '*' in inst_gen.other_params['look_ahead']):
                seed(inst_gen.s_rd_seed + 3)
                s_paths_p = CundiBoy.offer.gen_empiric_p_sp(inst_gen, hist_p, W_p)
                return hist_p, W_p, s_paths_p 

            else:
                return hist_p, W_p, None
        
        
        # Historic prices
        def gen_hist_p(inst_gen,rd_function,ex_p,**kwargs) -> dict[dict]:
            hist_p = {t:dict() for t in inst_gen.Horizon}
            if inst_gen.other_params['historical'] != False and ('p' in inst_gen.other_params['historical'] or '*' in inst_gen.other_params['historical']):
                hist_p[0] = {(i,k):[round(rd_function(ex_p[i,k]*(1-kwargs['r_f_params']),ex_p[i,k]*(1+kwargs['r_f_params'])),2) if i in inst_gen.M_kt[k,t] else 1000 for t in inst_gen.historical] for i in inst_gen.Suppliers for k in inst_gen.Products}
            else:
                hist_p[0] = {(i,k):[] for i in inst_gen.Suppliers for k in inst_gen.Products}

            return hist_p


        # Realized (real) prices
        def gen_W_p(inst_gen,rd_function,hist_p,ex_p,**kwargs) -> tuple:
            '''
            W_p: (dict) quantity of k \in K offered by supplier i \in M on t \in T
            '''
            W_p = dict()
            for t in inst_gen.Horizon:
                W_p[t] = dict()   
                for i in inst_gen.Suppliers:
                    for k in inst_gen.Products:
                        if i in inst_gen.M_kt[k,t]:
                            W_p[t][(i,k)] = round(rd_function(ex_p[i,k]*(1-kwargs['r_f_params']),ex_p[i,k]*(1+kwargs['r_f_params'])),2)
                        else:   W_p[t][(i,k)] = 10000

                        if t < inst_gen.T - 1:
                            hist_p[t+1][i,k] = hist_p[t][i,k] + [W_p[t][i,k]]

            return W_p, hist_p

