import numpy as np; import gurobipy as gu

class Inventory():
    @staticmethod
    def det_FIFO(purchase:dict,inst_gen,env) -> dict[float,int]:
        demand_compliance = {}
        for k in inst_gen.Products:
            left_to_comply = inst_gen.W_d[env.t][k]
            for o in range(inst_gen.O_k[k],0,-1):
                demand_compliance[k,o] = min(env.state[k,o], left_to_comply)
                left_to_comply -= demand_compliance[k,o]
            
            demand_compliance[k,0] = min(sum(purchase[i,k] for i in inst_gen.Suppliers), left_to_comply)
        
        return demand_compliance
    
    
    @staticmethod
    def Stochastic_Rolling_Horizon(state,env,inst_gen,objs={"costs":1},fixed_suppliers=None):

        # ----------------------------------------
        # MODEL PARAMETERS
        # ----------------------------------------

        # Initial inventory of products
        I_0 = state.copy()

        # Look ahead window
        Num_periods = inst_gen.sp_window_sizes[env.t]
        T = range(Num_periods)

        serv_level = inst_gen.theta

        # Sets
        M = inst_gen.Suppliers; K = inst_gen.Products; S = inst_gen.Samples

        # Routing cost approximation
        C_MIP = {i:inst_gen.c[0,i]+inst_gen.c[i,0] for i in inst_gen.Suppliers} 

        m = gu.Model('Inventory')

        # ----------------------------------------
        # MODEL VARIABLES
        # ----------------------------------------

        # How much to buy from supplier i of product k at time period t on scenario s 
        z = {(i,k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t,s))) for t in T for k in K for s in S for i in inst_gen.M_kt[k,env.t + t]}

        # 1 if supplier i is selected at time period t on scenario s, 0 otherwise
        w = {(i,t,s):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t,s))) for t in T for i in M for s in S}

        # Final inventory of product k aged o time periods at the warehouse by time period t on scenario s 
        ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,o,s))) for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S}

        # Sold units of product k aged o time periods, by time period t on scenario s
        y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_"+str((k,t,o,s))) for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S}

        # Backorder units of product k at time period t on scenario s
        bo = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,s))) for t in T for k in K for s in S}

        # Auxiliary variable for the objective function
        v = m.addVar(vtype=gu.GRB.CONTINUOUS,name="Objs_Aux_Var",obj=1)

        # ----------------------------------------
        # MODEL CONSTRAINTS
        # ----------------------------------------

        for s in S:
            ''' Inventory constraints '''
            for k in K:
                for t in T:
                    m.addConstr(ii[k,t,0,s] == gu.quicksum(z[i,k,t,s] for i in inst_gen.M_kt[(k,env.t + t)]) - y[k,t,0,s], f'Fresh produce inventory {k,t,s}')
                    
            for k in K:
                for o in inst_gen.Ages[k]:
                    m.addConstr(ii[k,0,o,s] == I_0[k,o] - y[k,0,o,s], f"Current time period inventory {k,o,s}")
                    
            for k in K:
                for t in T:
                    for o in inst_gen.Ages[k]:
                        if t > 0:
                            m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s] - y[k,t,o,s], f'Lookahead inventory {k,t,o,s}')

            ''' Demand compliance '''
            for k in K: 
                for t in T:
                    m.addConstr(gu.quicksum(y[k,t,o,s] for o in range(inst_gen.O_k[k] + 1)) + bo[k,t,s] == inst_gen.s_paths_d[env.t][t,s][k], f'Backorders {k,t,s}')   

            ''' Supply constraints '''
            for t in T:
                for k in K:
                    for i in inst_gen.M_kt[k,env.t + t]: 
                        m.addConstr(z[i,k,t,s] <= inst_gen.s_paths_q[env.t][t,s][i,k]*w[i,t,s], f'Purchase at supplier {i,k,t,s}')
            
            
            for t in T:
                for i in M:
                    m.addConstr(gu.quicksum(z[i,k,t,s] for k in K if (i,k,t,s) in z) <= inst_gen.Q*w[i,t,s], f'Vehicle capacity {i,t,s}')
                    min_q = min(sum(inst_gen.s_paths_q[env.t][t,s][i,k] for k in inst_gen.Products if i in inst_gen.M_kt[k,env.t+t]),inst_gen.rr*inst_gen.Q)
                    if t != 0:
                        m.addConstr(gu.quicksum(z[i,k,t,s] for k in K if (i,k,t,s) in z) >= min_q*w[i,t,s], f'Vehicle capacity {i,t,s}')

            '''' NON-ANTICIPATIVITY CONSTRAINTS '''
            # for k in K:
            #     for i in inst_gen.M_kt[k,env.t]:
            #         m.addConstr(z[i,k,0,s] == gu.quicksum(z[i,k,0,ss] for ss in S)/len(S), f'Anticipativity purchase {i,k,s}')

        ''' Service Level control constraint '''
        for k in K:
            m.addConstr(gu.quicksum(gu.quicksum(y[k,t,o,s] for t in T for o in range(inst_gen.O_k[k] + 1))/sum(inst_gen.s_paths_d[env.t][t,s][k] for t in T) for s in S) >= serv_level*len(inst_gen.Samples))

        ''' Fixed suppliers constraints '''
        if fixed_suppliers != None:
            for i in inst_gen.Suppliers:
                if i in inst_gen.Suppliers:
                    m.addConstr(gu.quicksum(w[i,0,s])==inst_gen.S,f'Fixed supplier{i,s}')
                else:
                    m.addConstr(gu.quicksum(w[i,0,s])==0,f'Fixed supplier{i,s}')
                    

        # ----------------------------------------
        # OBJECTIVE FUNCTION
        # ----------------------------------------

        ''' Expected costs '''
        purch_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.W_p[env.t][i,k]*z[i,k,t,s] for k in K for t in T for s in S for i in inst_gen.M_kt[k,env.t + t])/len(S)
        backo_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.back_o_cost[k]*bo[k,t,s] for k in K for t in T for s in S)/len(S)
        rout_aprox_cost = gu.quicksum((inst_gen.gamma**t)*C_MIP[i]*w[i,t,s] for i in M for t in T for s in S)/len(S)
        holding_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.W_h[env.t][k]*ii[k,t,o,s] for k in K for t in T for o in range(inst_gen.O_k[k]) for s in S)/len(S)
        
        if len(objs) == 1:
            if "costs" in objs:
                if inst_gen.hold_cost: m.addConstr(v >= purch_cost + backo_cost + rout_aprox_cost + holding_cost)
                else: m.addConstr(v >= purch_cost + backo_cost + rout_aprox_cost)
            else:
                e = list(objs.keys())[0]
                transp_aprox_impact = gu.quicksum((inst_gen.gamma**t)*(inst_gen.c_LCA[e][k][0,i]+inst_gen.c_LCA[e][k][i,0])*z[i,k,t,s] for k in K for t in T for s in S for i in inst_gen.M_kt[k,env.t + t])/len(S)
                storage_impact = gu.quicksum((inst_gen.gamma**t)*inst_gen.h_LCA[e][k]*ii[k,t,o,s] for k in K for t in T for o in range(inst_gen.O_k[k]) for s in S)/len(S)
                waste_impact = gu.quicksum((inst_gen.gamma**t)*inst_gen.waste_LCA[e][k]*ii[k,t,inst_gen.O_k[k],s] for k in K for t in T for s in S)/len(S)
                
                m.addConstr(v >= transp_aprox_impact + storage_impact + waste_impact)

        else:
            if inst_gen.hold_cost: m.addConstr((env.norm_matrix["costs"]["worst"] - env.norm_matrix["costs"]["best"])*v >= objs["costs"]*(purch_cost + backo_cost + rout_aprox_cost + holding_cost - env.norm_matrix["costs"]["best"]))
            else: m.addConstr((env.norm_matrix["costs"]["worst"] - env.norm_matrix["costs"]["best"])*v >= objs["costs"]*(purch_cost + backo_cost + rout_aprox_cost - env.norm_matrix["costs"]["best"]))
            for e in inst_gen.E:
                transp_aprox_impact = gu.quicksum((inst_gen.gamma**t)*(inst_gen.c_LCA[e][k][0,i]+inst_gen.c_LCA[e][k][i,0])*z[i,k,t,s] for k in K for t in T for s in S for i in inst_gen.M_kt[k,env.t + t])/len(S)
                storage_impact = gu.quicksum((inst_gen.gamma**t)*inst_gen.h_LCA[e][k]*ii[k,t,o,s] for k in K for t in T for o in range(inst_gen.O_k[k]) for s in S)/len(S)
                waste_impact = gu.quicksum((inst_gen.gamma**t)*inst_gen.waste_LCA[e][k]*ii[k,t,inst_gen.O_k[k],s] for k in K for t in T for s in S)/len(S)

                m.addConstr((env.norm_matrix[e]["worst"] - env.norm_matrix[e]["best"])*v >= objs[e]*(transp_aprox_impact+storage_impact+waste_impact-env.norm_matrix[e]["best"]))


        # ----------------------------------------
        # MODEL IMPLEMENTATION
        # ----------------------------------------

        m.update()
        m.setParam('OutputFlag',0)
        m.optimize()
        '''if m.Status == 3: # If model is infeasible, return IIS computation
            m.computeIIS()
            for const in m.getConstrs():
                if const.IISConstr:
                    print(const.ConstrName)'''

        if (not inst_gen.sustainability and len(objs) == 1) or (inst_gen.sustainability and len(objs) > 1):
            
            # ----------------------------------------
            # DECISIONS RETRIEVAL
            # ----------------------------------------

            # Purchase
            purchase = {(i,k): 0 for i in M for k in K}
            for k in K:
                for i in inst_gen.M_kt[k,env.t]:
                    purchase[i,k] = z[i,k,0,0].x
            
            demand_compliance = {}
            for k in K:
                
                # If fresh product is available
                if sum(purchase[i,k] for i in inst_gen.M_kt[k,env.t]) > 0:
                    demand_compliance[k,0] = 0
                    for s in S:
                        if y[k,0,0,s].x > 0:
                            demand_compliance[k,0] += 1
                    demand_compliance[k,0] /= len(S)
                
                else:
                    demand_compliance[k,0] = 1

                for o in range(1,inst_gen.O_k[k]+1):
                    demand_compliance[k,o] = 0
                    for s in S:
                        # If still some left to comply
                        if round(bo[k,0,s].x + sum(y[k,0,oo,s].x for oo in range(inst_gen.O_k[k],o,-1)),3) < round(inst_gen.s_paths_d[env.t][0,s][k],3):
                            #... and used to comply
                            if y[k,0,o,s].x > 0:
                                demand_compliance[k,o] += 1
                        else:
                            demand_compliance[k,o] += 1
                    demand_compliance[k,o] /= len(S)

            action = [purchase, demand_compliance]
            
            # ----------------------------------------
            # LOOKAHEAD DECISIONS
            # ----------------------------------------

            I0 = {t:{s:{(k,o):ii[k,t,o,s].x for k in K for o in range(inst_gen.O_k[k]+1)} for s in S} for t in T}
            zz = {t:{s:{(i,k):z[i,k,t,s].x for k in K for i in inst_gen.M_kt[k,env.t+t]} for s in S} for t in T}
            bb = {t:{s:{k:bo[k,t,s].x for k in K} for s in S} for t in T}
            yy = {t:{s:{(k,o):y[k,t,o,s].x for k in K for o in range(inst_gen.O_k[k]+1)} for s in S} for t in T}

            la_decisions = [I0, zz, bb, yy]

            return action,la_decisions

        else:
            
            # ----------------------------------------
            # Objectives Performance
            # ----------------------------------------

            impacts = dict()
            if inst_gen.hold_cost: impacts["costs"] = purch_cost.getValue() + backo_cost.getValue() + rout_aprox_cost.getValue() + holding_cost.getValue()
            else: impacts["costs"] = purch_cost.getValue() + backo_cost.getValue() + rout_aprox_cost.getValue()
            for e in inst_gen.E:
                transp_aprox_impact = gu.quicksum((inst_gen.c_LCA[e][k][0,i]+inst_gen.c_LCA[e][k][i,0])*z[i,k,t,s] for k in K for t in T for s in S for i in inst_gen.M_kt[k,env.t + t])/len(S)
                storage_impact = gu.quicksum(inst_gen.h_LCA[e][k]*ii[k,t,o,s] for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S)/len(S)
                waste_impact = gu.quicksum(inst_gen.waste_LCA[e][k]*ii[k,t,inst_gen.O_k[k],s] for k in K for t in T for s in S)/len(S)

                impacts[e] = transp_aprox_impact.getValue() + storage_impact.getValue() + waste_impact.getValue()
                #impacts[e] = transp_aprox_impact.getValue() + waste_impact.getValue()

            return impacts
