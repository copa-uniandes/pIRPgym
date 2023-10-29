import numpy as np

import gurobi as gu

from ..InstanceGenerator import instance_generator
from ..pIRPenv import steroid_IRP


class Inventory():
    @staticmethod
    def det_FIFO(state:dict,purchase:dict,inst_gen:instance_generator,env: steroid_IRP) -> dict[float]:
        demand_compliance = {}
        for k in inst_gen.Products:
            left_to_comply = inst_gen.W_d[env.t][k]
            for o in range(inst_gen.O_k[k],0,-1):
                demand_compliance[k,o] = min(env.state[k,o], left_to_comply)
                left_to_comply -= demand_compliance[k,o]
            
            demand_compliance[k,0] = min(sum(purchase[i,k] for i in inst_gen.Suppliers), left_to_comply)
        
        return demand_compliance
    
    
    @staticmethod
    def Stochastic_Rolling_Horizon(state,env,inst_gen):
        # State
        I_0 = state.copy()

        # Look ahead window
        Num_periods = inst_gen.sp_window_sizes[env.t]
        T = range(Num_periods)

        theta = 0.15
        if Num_periods == inst_gen.LA_horizon:
            theta *= 1.25
        elif Num_periods < inst_gen.LA_horizon and env.t < (inst_gen.T - 1):
            theta *= (1+0.25*(inst_gen.LA_horizon-Num_periods+1))
        else:
            theta = 1

        # Iterables
        M = inst_gen.Suppliers; K = inst_gen.Products; S = inst_gen.Samples

        # Initialization routing cost
        C_MIP = {(i,t):inst_gen.c[0,i]+inst_gen.c[i,0] for t in T for i in inst_gen.Suppliers} 

        m = gu.Model('Inventory')

        # Variables    
        # How much to buy from supplier i of product k at time t 
        z = {(i,k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t,s))) for t in T for k in K for s in S for i in inst_gen.M_kt[k,env.t + t]}
        tuples = [(i,k,t,s) for t in T for k in K for s in S for i in inst_gen.M_kt[k,env.t + t]]

        # 1 if supplier i is selected at time t, 0 otherwise
        w = {(i,t,s):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t,s))) for t in T for i in M for s in S}

        # Final inventory of product k of old o at time t 
        ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,o,s))) for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S}

        # Units sold of product k at time t of old age o
        y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_"+str((k,t,o,s))) for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S}

        # Units in backorders of product k at time t
        bo = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,s))) for t in T for k in K for s in S}

        boo = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="boo_"+str((k,t,s))) for t in T for k in K for s in S}

        for s in S:
            ''' Inventory constraints '''
            for k in K:
                for t in T:
                    m.addConstr(ii[k,t,0,s] == gu.quicksum(z[i,k,t,s] for i in inst_gen.M_kt[(k,env.t + t)]) - y[k,t,0,s], str(f'Inventario edad 0 {k}{t}{s}'))
                    
            for k in K:
                for o in inst_gen.Ages[k]:
                    m.addConstr(ii[k,0,o,s] == I_0[k,o] - y[k,0,o,s], str(f'Inventario periodo 0 k = {k}, o = {o}, s = {s}'))
                    
            for k in K:
                for t in T:
                    for o in inst_gen.Ages[k]:
                        if t > 0:
                            m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s] - y[k,t,o,s], str(f'Inventario {k}{t}{o}{s}'))

            for k in K: 
                for t in T:
                    m.addConstr(gu.quicksum(y[k,t,o,s] for o in range(inst_gen.O_k[k] + 1)) + bo[k,t,s] == inst_gen.s_paths_d[env.t][t,s][k], f'backorders {k}{t}{s}')   


            ''' Purchase constraints '''
            for t in T:
                for k in K:
                    for i in inst_gen.M_kt[k,env.t + t]: 
                        m.addConstr(z[i,k,t,s] <= inst_gen.s_paths_q[env.t][t,s][i,k]*w[i,t,s], f'Purchase {i}{k}{t}{s}')
                        
            for t in T:
                for i in M:
                    m.addConstr(gu.quicksum( z[i,k,t,s] for k in K if (i,k,t,s) in z) <= inst_gen.Q, f'Vehicle capacity {i}{t}{s}')
        
            '''' NON-ANTICIPATIVITY CONSTRAINTS '''
            for k in K:

                for i in inst_gen.M_kt[k,env.t]:
                    m.addConstr(z[i,k,0,s] == gu.quicksum(z[i,k,0,ss] for ss in S)/len(S), f'Anticipativity purchase {i}{k}{s}')
                
            for i in M:
                m.addConstr(w[i,0,s] == gu.quicksum(w[i,0,ss] for ss in S)/len(S), f'Anticipativity binary {i}{s}')

        ''' Backorders control restriction '''        
        m.addConstr(gu.quicksum(bo[k,t,s] for t in T for k in K for s in S) <= theta*sum(inst_gen.s_paths_d[env.t][t,s][k] for t in T for k in K for s in S) + gu.quicksum(boo[k,t,s] for t in T for k in K for s in S))
        
        ''' Perished products restriction '''
        for t in T:
            m.addConstr(gu.quicksum(ii[k,t,inst_gen.O_k[k],s] for k in K for s in S) <= 0.1*sum(inst_gen.s_paths_d[env.t][t,s][k] for k in K for s in S))

        compra = gu.quicksum(inst_gen.W_p[env.t][i,k]*z[i,k,t,s] for k in K for t in T for s in S for i in inst_gen.M_kt[k,env.t + t])/len(S) + \
            inst_gen.back_o_cost*gu.quicksum(bo[k,t,s] for k in K for t in T for s in S)/len(S) + 6e9*gu.quicksum(boo[k,t,s] for k in K for t in T for s in S)/len(S)
        
        ruta = gu.quicksum(C_MIP[i,t]*w[i,t,s] for i in M for t in T for s in S)

        ingresos = gu.quicksum(inst_gen.sell_prices[k,o]*y[k,t,o,s] for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S)

        m.setObjective(ingresos - compra - ruta, gu.GRB.MAXIMIZE)
        
        m.update()
        m.setParam('OutputFlag',0)
        m.optimize()
        if m.Status == 3:
            m.computeIIS()
            for const in m.getConstrs():
                if const.IISConstr:
                    print(const.ConstrName)

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

        
        # Back-orders
        double_check = {(k,t): bo[k,0,0].x for k in K}

        action = [purchase, demand_compliance]
        
        I0 = {}
        for t in T: 
            I0[t] = {}
            for s in S:  
                I0[t][s] = {}
                for k in K:
                    for o in range(inst_gen.O_k[k]+1):
                        I0[t][s][k,o] = ii[k,t,o,s].x
        zz = {}
        for t in T:
            zz[t] = {}
            for s in S:
                zz[t][s] = {}
                for k in K:
                    for i in inst_gen.M_kt[k,env.t + t]:
                        zz[t][s][i,k] = z[i,k,t,s].x 
        bb = {}
        for t in T:
            bb[t] = {}
            for s in S:
                bb[t][s] = {}
                for k in K:
                    bb[t][s][k] = bo[k,t,s].x
        
        yy = {}
        for t in T:
            yy[t] = {}
            for s in S:
                yy[t][s] = {}
                for k in K:
                    for o in range(inst_gen.O_k[k]+1):
                        yy[t][s][k,o] = y[k,t,o,s].x

        la_decisions = [I0, zz, bb, yy]

        return action, la_decisions


    @staticmethod
    def Stochastic_RH_Age_Demand(state,env,inst_gen):

        solucionTTP = {0:[  np.zeros(inst_gen.M+1, dtype=bool), 
                                np.zeros(inst_gen.M+1, dtype=int), 
                                np.zeros((inst_gen.M+1, inst_gen.K), dtype=bool), 
                                np.zeros((inst_gen.M+1, inst_gen.K), dtype=int), 
                                np.full(inst_gen.M+1, -1, dtype = int), 
                                np.zeros(inst_gen.M+1, dtype=int), 
                                np.zeros(inst_gen.K, dtype=int), 0, 0]}

        # State
        I_0 = state.copy()

        # Look ahead window
        Num_periods = inst_gen.sp_window_sizes[env.t]
        T = range(Num_periods)

        theta = 0.15
        if Num_periods == inst_gen.LA_horizon:
            theta *= 1.25
        elif Num_periods < inst_gen.LA_horizon and env.t < (inst_gen.T - 1):
            theta *= (1+0.25*(inst_gen.LA_horizon-Num_periods+1))
        else:
            theta = 1

        # Iterables
        M = inst_gen.Suppliers; K = inst_gen.Products; S = inst_gen.Samples

        # Initialization routing cost
        C_MIP = {(i,t):inst_gen.c[0,i]+inst_gen.c[i,0] for t in T for i in inst_gen.Suppliers} 

        m = gu.Model('Inventory')

        # Variables    
        # How much to buy from supplier i of product k at time t 
        z = {(i,k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t,s))) for t in T for k in K for s in S for i in inst_gen.M_kt[k,env.t + t]}

        # 1 if supplier i is selected at time t, 0 otherwise
        w = {(i,t,s):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t,s))) for t in T for i in M for s in S}

        # Final inventory of product k of old o at time t 
        ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,o,s))) for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S}

        # Units sold of product k at time t of old age o
        y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_"+str((k,t,o,s))) for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S}

        # Units in backorders of product k at time t
        bo = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,o,s))) for t in T for k in K for o in range(inst_gen.O_k[k] + 1) for s in S}

        boo = m.addVar(vtype=gu.GRB.CONTINUOUS, name="boo")

        for s in S:
            ''' Inventory constraints '''
            for k in K:
                for t in T:
                    m.addConstr(ii[k,t,0,s] == gu.quicksum(z[i,k,t,s] for i in inst_gen.M_kt[(k,env.t + t)]) - y[k,t,0,s], str(f'Inventario edad 0 {k}{t}{s}'))
                    
            for k in K:
                for o in inst_gen.Ages[k]:
                    m.addConstr(ii[k,0,o,s] == I_0[k,o] - y[k,0,o,s], str(f'Inventario periodo 0 k = {k}, o = {o}, s = {s}'))
                    
            for k in K:
                for t in T:
                    for o in inst_gen.Ages[k]:
                        if t > 0:
                            m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s] - y[k,t,o,s], str(f'Inventario {k}{t}{o}{s}'))

            for k in K: 
                for t in T:
                    for o in range(inst_gen.O_k[k] + 1):
                        m.addConstr(y[k,t,o,s]  + bo[k,t,o,s] == inst_gen.s_paths_d[env.t][t,s][k,o], f'backorders {k}{t}{o}{s}')   


            ''' Purchase constraints '''
            for t in T:
                for k in K:
                    for i in inst_gen.M_kt[k,env.t + t]: 
                        m.addConstr(z[i,k,t,s] <= inst_gen.s_paths_q[env.t][t,s][i,k]*w[i,t,s], f'Purchase {i}{k}{t}{s}')
                        
            #for t in T:
            #    for i in M:
            #        m.addConstr(gu.quicksum( z[i,k,t,s] for k in K if (i,k,t,s) in z) <= inst_gen.Q, f'Vehicle capacity {i}{t}{s}')
        
            '''' NON-ANTICIPATIVITY CONSTRAINTS '''
            for k in K:

                for i in inst_gen.M_kt[k,env.t]:
                    m.addConstr(z[i,k,0,s] == gu.quicksum(z[i,k,0,ss] for ss in S)/len(S), f'Anticipativity purchase {i}{k}{s}')
                
            for i in M:
                m.addConstr(w[i,0,s] == gu.quicksum(w[i,0,ss] for ss in S)/len(S), f'Anticipativity binary {i}{s}')

        ''' Backorders control restriction '''        
        m.addConstr(gu.quicksum(bo[k,t,o,s] for o in range(inst_gen.O_k[k]+1) for t in T for k in K for s in S) <= theta*sum(inst_gen.s_paths_d[env.t][t,s][k,o] for o in range(inst_gen.O_k[k]+1) for t in T for k in K for s in S) + boo)
        
        compra = gu.quicksum(inst_gen.W_p[env.t][i,k]*z[i,k,t,s] for k in K for t in T for s in S for i in inst_gen.M_kt[k,env.t + t])/len(S) + \
            inst_gen.back_o_cost*gu.quicksum(bo[k,t,o,s] for o in range(inst_gen.O_k[k]+1) for k in K for t in T for s in S)/len(S) + 6e9*boo/len(S)
        
        ruta = gu.quicksum(C_MIP[i,t]*w[i,t,s] for i in M for t in T for s in S)

        ingresos = gu.quicksum(inst_gen.sell_prices[k,o]*y[k,t,o,s] for k in K for t in T for o in range(inst_gen.O_k[k] + 1) for s in S)

        m.setObjective(ingresos - compra - ruta, gu.GRB.MAXIMIZE)
        
        m.update()
        m.setParam('OutputFlag',0)
        m.optimize()
        if m.Status == 3:
            m.computeIIS()
            for const in m.getConstrs():
                if const.IISConstr:
                    print(const.ConstrName)

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
                    # If wasn't able to comply
                    if bo[k,0,o,s].x > 0:
                        #... and indeed didn't have available product
                        if ii[k,0,o,s].x == 0:
                            demand_compliance[k,o] += 1
                    else:
                        demand_compliance[k,o] += 1
                demand_compliance[k,o] /= len(S)

        
        rutas = []

        action = [rutas, purchase, demand_compliance]
        
        I0 = {}
        for t in T: 
            I0[t] = {}
            for s in S:  
                I0[t][s] = {}
                for k in K:
                    for o in range(inst_gen.O_k[k]+1):
                        I0[t][s][k,o] = ii[k,t,o,s].x
        zz = {}
        for t in T:
            zz[t] = {}
            for s in S:
                zz[t][s] = {}
                for k in K:
                    for i in inst_gen.M_kt[k,env.t + t]:
                        zz[t][s][i,k] = z[i,k,t,s].x 
        bb = {}
        for t in T:
            bb[t] = {}
            for s in S:
                bb[t][s] = {}
                for k in K:
                    for o in range(inst_gen.O_k[k]+1):
                        bb[t][s][k,o] = bo[k,t,o,s].x
        
        yy = {}
        for t in T:
            yy[t] = {}
            for s in S:
                yy[t][s] = {}
                for k in K:
                    for o in range(inst_gen.O_k[k]+1):
                        yy[t][s][k,o] = y[k,t,o,s].x

        la_decisions = [I0, zz, bb, yy]

        return action, la_decisions

