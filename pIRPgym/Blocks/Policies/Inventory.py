import numpy as np; import gurobipy as gu
from itertools import combinations

class Inventory():
    @staticmethod
    def det_FIFO(state:dict,purchase:dict,inst_gen,env) -> dict[float,int]:
        demand_compliance = {}
        for k in inst_gen.Products:
            left_to_comply = inst_gen.W_d[env.t][k]
            for o in range(inst_gen.O_k[k],0,-1):
                demand_compliance[k,o] = min(env.state[k,o], left_to_comply)
                left_to_comply -= demand_compliance[k,o]
            
            demand_compliance[k,0] = min(sum(purchase[i,k] for i in inst_gen.Suppliers), left_to_comply)
        
        return demand_compliance
    
    
    @staticmethod
    def Stochastic_Rolling_Horizon(state,env,inst_gen,objs={"costs":1}):

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
            for k in K:

                for i in inst_gen.M_kt[k,env.t]:
                    m.addConstr(z[i,k,0,s] == gu.quicksum(z[i,k,0,ss] for ss in S)/len(S), f'Anticipativity purchase {i,k,s}')

        ''' Service Level control constraint '''
        for k in K:
            m.addConstr(gu.quicksum(gu.quicksum(y[k,t,o,s] for t in T for o in range(inst_gen.O_k[k] + 1))/sum(inst_gen.s_paths_d[env.t][t,s][k] for t in T) for s in S) >= serv_level*len(inst_gen.Samples))

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

            return action, la_decisions

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
    


    class IRP():
        
        @staticmethod
        def get_iterables(env, inst_gen):
            
            S = inst_gen.Samples; P = inst_gen.Products; N = inst_gen.Suppliers
            A = inst_gen.A; V = inst_gen.V
            T = range(inst_gen.sp_window_sizes[env.t])
            
            M = list()
            for r in range(2,len(N)+1):
                M += combinations(iterable = N, r = r)
            
            return S, T, P, N, A, V, M

        @staticmethod
        def build_model(state, env, inst_gen):
            
            ###################################
            # Parameters Setting
            ###################################

            I_0 = state.copy()
            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)
            
            m = gu.Model("VRP")

            ###################################
            # Variables
            ###################################

            ''' Whether supplier i in N is visited on day t in T on sample path s in S '''
            z = {(i,t,s):m.addVar(vtype=gu.GRB.BINARY, name=f"z_{i,t,s}") for i in N for t in T for s in S}; z.update({(0,t,s):m.addVar(vtype=gu.GRB.INTEGER, name=f"z_{0,t,s}") for t in T for s in S})
            
            ''' Number of times that the arc (i,j) in A is traversed on day t in T on sample path s in S '''
            x = {(i,j,t,s):m.addVar(vtype=gu.GRB.BINARY, name=f"x_{i,j,t,s}") for (i,j) in A for t in T for s in S}
            
            ''' Amount of product p in P that traverses arc (i,j) in A on day t in T on sample path s in S '''
            y = {(i,j,p,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"y_{i,j,p,t,s}") for (i,j) in A for p in P for t in T for s in S}

            ''' Amount of product p in P bought from supplier i in N_p on day t in T on sample path s in S '''
            q = {(i,p,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"q_{i,p,t,s}") for t in T for p in P for i in inst_gen.M_kt[p,env.t + t] for s in S}

            ''' Inventory level of product p in P that has aged o in O_p days at the warehouse by the end of day t in T on sample path s in S '''
            I = {(p,o,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"I_{p,o,t,s}") for p in P for o in range(inst_gen.O_k[p]+1) for t in T for s in S}
            
            ''' Amount of product p in P that has aged o in O_p days at the warehouse used to fulfill the customer's demand on day t in T on sample path s in S '''
            v = {(p,o,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"v_{p,o,t,s}") for p in P for o in range(inst_gen.O_k[p]+1) for t in T for s in S}
            
            ''' Unfulfilled demand of product p in P on day t in T on sample path s in S '''
            b = {(p,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"b_{p,t,s}") for p in P for t in T for s in S}

            ''' Auxiliary Variable for objectives '''
            Z = m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"ZZ", obj = 1)

            ###################################
            # Constraints
            ###################################

            for s in S:

                for p in P:
                    for o in range(1,inst_gen.O_k[p]+1):
                        ''' Initial inventory of aged product '''
                        m.addConstr(I[p,o,0,s] == I_0[p,o] - v[p,o,0,s])
                    
                    for t in T:
                        ''' Inventory of fresh product '''
                        m.addConstr(I[p,0,t,s] == gu.quicksum(q[i,p,t,s] for i in inst_gen.M_kt[p,env.t + t]) - v[p,0,t,s])
                        
                        if t >= 1:
                            for o in range(1,inst_gen.O_k[p]+1):
                                ''' Inventory dynamics throughout the lookahead horizon '''
                                m.addConstr(I[p,o,t,s] == I[p,o-1,t-1,s] - v[p,o,t,s])

                for t in T:
                    
                    ''' Fleet size '''
                    m.addConstr(z[0,t,s] <= inst_gen.F)

                    for i in N:
                        ''' Maximum allowed total purchase at each visited supplier '''
                        m.addConstr(gu.quicksum(q[i,p,t,s] for p in P if i in inst_gen.M_kt[p,env.t + t]) <= inst_gen.Q*z[i,t,s])
                        
                        #if t >= 1:
                        ''' Minimum required purchase at each visited supplier '''
                        m.addConstr(gu.quicksum(q[i,p,t,s] for p in P if i in inst_gen.M_kt[p,env.t + t]) >= inst_gen.rr*inst_gen.Q*z[i,t,s])
                        
                        for p in P:
                            if i in inst_gen.M_kt[p,env.t + t]:

                                ''' Product availability at each supplier '''
                                m.addConstr(q[i,p,t,s] <= inst_gen.s_paths_q[env.t][t,s][i,p]*z[i,t,s])
                                
                                ''' Flow of products at the supplier nodes '''
                                m.addConstr(gu.quicksum(y[i,j,p,t,s] for j in V if (i,j) in A) - gu.quicksum(y[j,i,p,t,s] for j in V if (j,i) in A) == q[i,p,t,s])
                            else:
                                m.addConstr(gu.quicksum(y[i,j,p,t,s] for j in V if (i,j) in A) - gu.quicksum(y[j,i,p,t,s] for j in V if (j,i) in A) == 0)
                        
                    for p in P:
                        
                        ''' Demand compliance and backorders '''
                        m.addConstr(gu.quicksum(v[p,o,t,s] for o in range(inst_gen.O_k[p]+1)) + b[p,t,s] == inst_gen.s_paths_d[env.t][t,s][p])
                        
                        ''' Flow of products at the warehouse '''
                        m.addConstr(gu.quicksum(y[i,0,p,t,s] for i in N) == gu.quicksum(q[i,p,t,s] for i in inst_gen.M_kt[p,env.t + t]))
                        m.addConstr(gu.quicksum(y[0,i,p,t,s] for i in N) == 0)
                        
                        ''' Flow variables association '''
                        for (i,j) in A:
                            m.addConstr(y[i,j,p,t,s] <= inst_gen.Q*x[i,j,t,s])

                    for i in V:
                        ''' Flow through the network '''
                        m.addConstr(gu.quicksum(x[i,j,t,s] for j in V if (i,j) in A) == z[i,t,s])
                        m.addConstr(gu.quicksum(x[j,i,t,s] for j in V if (j,i) in A) == z[i,t,s])
                    
                    for mm in M:
                        ''' Subtour elimination and capacity constraints '''
                        m.addConstr(inst_gen.Q*gu.quicksum(x[i,j,t,s] for (i,j) in A if (i in mm) and (j in mm)) <= gu.quicksum(inst_gen.Q*z[i,t,s] - gu.quicksum(q[i,p,t,s] for p in P if i in inst_gen.M_kt[p,env.t + t]) for i in mm))
                    
                    ''' Non-anticipativity constraints '''
                    for p in P:
                        for i in inst_gen.M_kt[p,env.t + t]:
                            m.addConstr(q[i,p,0,s] == gu.quicksum(q[i,p,0,ss] for ss in S)/len(S))
                    
                    for (i,j) in A:
                        m.addConstr(x[i,j,0,s] == gu.quicksum(x[i,j,0,ss] for ss in S)/len(S))
                        
            
            ''' Service Level requirement constraints '''
            for p in P:
                m.addConstr(gu.quicksum(gu.quicksum(v[p,o,t,s] for o in range(inst_gen.O_k[p]+1) for t in T)/sum(inst_gen.s_paths_d[env.t][t,s][p] for t in T) for s in S) >= inst_gen.theta*len(S))

            ''' Expected costs '''
            purch_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.W_p[env.t][i,p]*q[i,p,t,s] for p in P for t in T for s in S for i in inst_gen.M_kt[p,env.t + t])/len(S)
            backo_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.back_o_cost[p]*b[p,t,s] for p in P for t in T for s in S)/len(S)
            rout_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.c[i,j]*x[i,j,t,s] for (i,j) in A for t in T for s in S)/len(S)
            holding_cost = gu.quicksum((inst_gen.gamma**t)*inst_gen.W_h[env.t][p]*I[p,o,t,s] for p in P for t in T for o in range(inst_gen.O_k[p]) for s in S)/len(S)

            costs = (purch_cost,rout_cost,holding_cost,backo_cost) if inst_gen.hold_cost else (purch_cost,rout_cost,backo_cost)
            
            ''' Expected environmental impacts '''
            transp_impact, storage_impact, waste_impact = dict(), dict(), dict()
            for e in inst_gen.E:
                transp_impact[e] = gu.quicksum((inst_gen.gamma**t)*(inst_gen.c_LCA[e][p][i,j])*y[i,j,p,t,s] for p in P for (i,j) in A for t in T for s in S)/len(S)
                storage_impact[e] = gu.quicksum((inst_gen.gamma**t)*inst_gen.h_LCA[e][p]*I[p,o,t,s] for p in P for t in T for o in range(inst_gen.O_k[p]) for s in S)/len(S)
                waste_impact[e] = gu.quicksum((inst_gen.gamma**t)*inst_gen.waste_LCA[e][p]*I[p,inst_gen.O_k[p],t,s] for p in P for t in T for s in S)/len(S)
            
            m.setParam("OutputFlag",0)
            m.setParam("MIPGap",1e-5)

            return m, (z, x, y, q, I, v, b, Z), costs, (transp_impact, storage_impact, waste_impact)

        @staticmethod
        def get_routes(x, z, N, A, S, T):

            xx = {t:{s:{(i,j):0 for (i,j) in A} for s in S} for t in T}; routes = {t:{s:list() for s in S} for t in T}
            for t in T:
                for s in S:

                    m = int(round(z[0,t,s].X,2))
                    nodes_to_visit = N + []
                    for r in range(m):
                        nodes_to_visit += [0]
                        next_node = [j for j in nodes_to_visit if ((0,j) in A) and (x[0,j,t,s].X > 0.5)][0]
                        route = [0, next_node]; nodes_to_visit.remove(next_node)
                        while next_node != 0:
                            next_node = [j for j in nodes_to_visit if ((next_node,j) in A) and (x[next_node,j,t,s].X > 0.5)][0]
                            route += [next_node]; nodes_to_visit.remove(next_node)

                        arcs = [(route[ix],route[ix+1]) for ix in range(len(route)-1)]
                        for a in arcs: xx[t][s][a] = 1

                        routes[t][s] += [route]

            return xx, routes

        @staticmethod
        def get_demand_compliance(env, inst_gen, purchase, v, b, P, S):
            demand_compliance = dict()
            for p in P:
                
                # If fresh product is available, count the sample paths that used it
                if sum(purchase[i,p] for i in inst_gen.M_kt[p,env.t]) > 0:
                    demand_compliance[p,0] = sum([1 for s in S if v[p,0,0,s].X > 0])/len(S)
                # If no fresh product is available, allow for compliance
                else:
                    demand_compliance[p,0] = 1

                for o in range(1,inst_gen.O_k[p]+1):
                    # Count the sample paths in which needed to comply and used it or didn't need to comply
                    demand_compliance[p,o] = sum([1 if round(b[p,0,s].x + sum(v[p,oo,0,s].x for oo in range(inst_gen.O_k[p],o,-1)),3) >= round(inst_gen.s_paths_d[env.t][0,s][p],3) else (1 if v[p,o,0,s].X > 0 else 0) for s in S])/len(S)
            
            return demand_compliance

        @staticmethod
        def retrieve_decisions(env, inst_gen, q, x, z, v, b, I):

            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)

            xx, routes = Inventory.IRP.get_routes(x, z, N, A, S, T)

            purchase = {(i,p):q[i,p,0,0].x if i in inst_gen.M_kt[p,env.t] else 0 for i in N for p in P}
            routing = routes[0][0]
            demand_compliance = Inventory.IRP.get_demand_compliance(env, inst_gen, purchase, v, b, P, S)

            action = {'routing':routing,'purchase':purchase,'demand_compliance':demand_compliance}

            II = {t:{s:{(p,o):I[p,o,t,s].x for p in P for o in range(inst_gen.O_k[p]+1)} for s in S} for t in T}
            qq = {t:{s:{(i,p):q[i,p,t,s].x for p in P for i in inst_gen.M_kt[p,env.t+t]} for s in S} for t in T}
            bb = {t:{s:{p:b[p,t,s].x for p in P} for s in S} for t in T}
            vv = {t:{s:{(p,o):v[p,o,t,s].x for p in P for o in range(inst_gen.O_k[p]+1)} for s in S} for t in T}

            la_decisions = [II, qq, bb, vv]

            return action, la_decisions
        
        class verbose():

            @staticmethod
            def print_IRP_decisions(inst_gen,env,string,S,P,T,A,q,v,I,b,x,y):

                print("---------------------------------------------------------------------------------------")
                print(f"- OPTIMIZING {string} ")
                print("---------------------------------------------------------------------------------------")
                for s in S:
                    print(f"Sample path {s}")
                    print(f"\tPurchasing decisions:")
                    for p in P:
                        print(f"\t\tP{p} demand:", *[f"{round(inst_gen.s_paths_d[env.t][t,s][p],2)}" for t in T], sep="\t")
                        for i in inst_gen.M_kt[p,env.t+0]:
                            print(f"\t\t\tSupplier {i}:",[f"{round(q[i,p,t,s].X,2)}" for t in T],sep="\t")
                    
                    print(f"\tInventory Management:")
                    for p in P:
                        print(f"\t\tProduct {p}")
                        print(f"\t\t\tFulfil.",*[f"{[round(v[p,o,t,s].X,2) for o in range(inst_gen.O_k[p])]}" for t in T],sep="\t")
                        print(f"\t\t\tInvent.",*[f"{[round(I[p,o,t,s].X,2) for o in range(inst_gen.O_k[p])]}" for t in T],sep="\t")
                        print("\t\t\tPerished",*[f"\t{round(I[p,inst_gen.O_k[p],t,s].X,2)}" for t in T],sep="\t")
                        print("\t\t\tBackord.",*[f"\t{round(b[p,t,s].X,2)}" for t in T],sep="\t")
                    
                    print("\tRouting:")
                    for t in T:
                        print(f"\t\tDay {t}",[f"{(i,j),(round(y[i,j,1,t,s].X,2),round(y[i,j,2,t,s].X,2))}" for (i,j) in A if x[i,j,t,s].X > 0.5],sep="\t")
                
                print("\n")
            
            @staticmethod
            def print_costs_performance(costs, h):

                if h: (purch_cost,rout_cost,holding_cost,backo_cost) = costs
                else: (purch_cost,rout_cost,backo_cost) = costs; holding_cost = gu.LinExpr(0)

                print(f"Costs Performance")
                print("\tPurchase", round(purch_cost.getValue(),2))
                print("\tRouting", round(rout_cost.getValue(),2))
                if h:print("\tHolding", round(holding_cost.getValue(),2))
                print("\tBackorders", round(backo_cost.getValue(),2))
                print("\tTotal Cost", round(purch_cost.getValue() + rout_cost.getValue() + holding_cost.getValue() + backo_cost.getValue(), 2))
                print("\n")
            
            @staticmethod
            def print_environmental_performance(transp_impact, storage_impact, waste_impact, E):

                print(f"Environmental Performance")
                print("\t       ",*[" "*(9-len(e))+e for e in E],sep="\t")
                print("\tTransp.",*[" "*(9-len(str(round(transp_impact[e].getValue(),2))))+f"{round(transp_impact[e].getValue(),2)}" for e in E],sep="\t")
                print("\tStorage",*[" "*(9-len(str(round(storage_impact[e].getValue(),2))))+f"{round(storage_impact[e].getValue(),2)}" for e in E],sep="\t")
                print("\tWaste  ",*[" "*(9-len(str(round(waste_impact[e].getValue(),2))))+f"{round(waste_impact[e].getValue(),2)}" for e in E],sep="\t")
                print("\tTotal  ",*[" "*(9-len(str(round(transp_impact[e].getValue()+storage_impact[e].getValue()+waste_impact[e].getValue(),2))))+f"{round(transp_impact[e].getValue()+storage_impact[e].getValue()+waste_impact[e].getValue(),2)}" for e in E],sep="\t")
                print("\n")


    @staticmethod
    def optimize_environmental_indicator(e, m, impacts, state, env, inst_gen, verbose = False, action = False):

        ''' Minimize individual environmental indicator '''
        m.setObjective(gu.quicksum(im[e] for im in impacts))
        m.update(); m.optimize()

        z_e = m.objVal

        if verbose:
            transp_impact, storage_impact, waste_impact = impacts
            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)
            Inventory.IRP.verbose.print_environmental_performance(transp_impact, storage_impact, waste_impact, inst_gen.E)

        ''' Find cost-efficient solution '''
        m1, (zz, xx, yy, qq, II, vv, bb, ZZ), costs, (transp_impact, storage_impact, waste_impact) = Inventory.IRP.build_model(state, env, inst_gen)
        m1.addConstr(transp_impact[e] + storage_impact[e] + waste_impact[e] <= z_e)
        m1.setObjective(gu.quicksum(c for c in costs))
        m1.update(); m1.optimize()

        if verbose:
            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)
            Inventory.IRP.verbose.print_IRP_decisions(inst_gen, env, e, S, P, T, A, qq, vv, II, bb, xx, yy)
            Inventory.IRP.verbose.print_costs_performance(costs, inst_gen.hold_cost)
            Inventory.IRP.verbose.print_environmental_performance(transp_impact, storage_impact, waste_impact, inst_gen.E)

        if not action:

            ''' Compute metrics performances '''
            performances = dict()
            performances["costs"] = sum(c.getValue() for c in costs)
            for ee in inst_gen.E:
                performances[ee] = transp_impact[ee].getValue() + storage_impact[e].getValue() + waste_impact[e].getValue()
            
            return performances

        else:
            
            ''' Retrieve decisions '''
            actions, la_decisions = Inventory.IRP.retrieve_decisions(env, inst_gen, qq, xx, zz, vv, bb, II)

            return actions, la_decisions
        
    @staticmethod
    def optimize_costs(e, m, costs, impacts, state, env, inst_gen, verbose = False, action = False):

        ''' Minimize individual environmental indicator '''
        m.setObjective(gu.quicksum(c for c in costs))
        m.update(); m.optimize()

        z_c = m.objVal

        if verbose:
            transp_impact, storage_impact, waste_impact = impacts
            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)
            Inventory.IRP.verbose.print_environmental_performance(transp_impact, storage_impact, waste_impact, inst_gen.E)

        ''' Find cost-efficient solution '''
        m1, (zz, xx, yy, qq, II, vv, bb, ZZ), costs1, (transp_impact, storage_impact, waste_impact) = Inventory.IRP.build_model(state, env, inst_gen)
        m1.addConstr(gu.quicksum(c for c in costs1) <= z_c)
        m1.setObjective(transp_impact[e] + storage_impact[e] + waste_impact[e])
        m1.update(); m1.optimize()

        if verbose:
            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)
            Inventory.IRP.verbose.print_IRP_decisions(inst_gen, env, e, S, P, T, A, qq, vv, II, bb, xx, yy)
            Inventory.IRP.verbose.print_costs_performance(costs, inst_gen.hold_cost)
            Inventory.IRP.verbose.print_environmental_performance(transp_impact, storage_impact, waste_impact, inst_gen.E)

        print(f"\t\te", *[f"{(round(sum(im[ee].getValue() for im in impacts),4),round((transp_impact[ee].getValue()+storage_impact[ee].getValue()+waste_impact[ee].getValue(),4)))}" for ee in ["climate","water","land","fossil"]], sep="\t")
        
        if not action:

            ''' Compute metrics performances '''
            performances = dict()
            performances["costs"] = sum(c.getValue() for c in costs)
            for ee in inst_gen.E:
                performances[ee] = transp_impact[ee].getValue() + storage_impact[e].getValue() + waste_impact[e].getValue()
            
            return performances

        else:
            
            ''' Retrieve decisions '''
            actions, la_decisions = Inventory.IRP.retrieve_decisions(env, inst_gen, qq, xx, zz, vv, bb, II)

            return actions, la_decisions
            
    @staticmethod
    def Stochastic_RH_IRP(state,env,inst_gen,objs={"costs":1},verbose=False):
        
        ###################################
        # Model building
        ###################################
        m, (z, x, y, q, I, v, b, Z), costs, (transp_impact, storage_impact, waste_impact) = Inventory.IRP.build_model(state, env, inst_gen)
        
        m.addConstr((env.norm_matrix["costs"]["worst"] - env.norm_matrix["costs"]["best"])*Z >= objs["costs"]*(gu.quicksum(c for c in costs) - env.norm_matrix["costs"]["best"]))
        for e in inst_gen.E:
            m.addConstr((env.norm_matrix[e]["worst"] - env.norm_matrix[e]["best"])*Z >= objs[e]*(storage_impact[e]+waste_impact[e]+transp_impact[e] - env.norm_matrix[e]["best"]))

        m.update()
        m.optimize()

        if verbose:
            S, T, P, N, A, V, M = Inventory.IRP.get_iterables(env, inst_gen)
            Inventory.IRP.verbose.print_IRP_decisions(inst_gen, env, list(objs.values()), S, P, T, A, q, v, I, b, x, y)
            Inventory.IRP.verbose.print_costs_performance(costs, inst_gen.hold_cost)
            Inventory.IRP.verbose.print_environmental_performance(transp_impact, storage_impact, waste_impact, inst_gen.E)

        ###################################
        # Decisions retrieval
        ###################################

        actions, la_decisions = Inventory.IRP.retrieve_decisions(env, inst_gen, q, x, z, v, b, I)

        return actions, la_decisions


