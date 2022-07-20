# -*- coding: utf-8 -*-
"""

"""

import gurobipy as gu
import numpy as np
import matplotlib.pyplot as plt
import time
from random import random, seed, randint, shuffle
import imageio
path1 = "C:/Users/juan_/OneDrive - Universidad de los Andes/1. MIIND/Dani/"

seed(20)

#%% Instance Generator

#This is the instance setting
Vertex = 10
Products = 10
Periods = 7
lambda1 = 0.5
replica = 2
Historical = 40

V = range(Vertex) # Nodes (vertices) in network: suppliers + warehouse (0)
M = range(1,Vertex) # Set of suppliers
T = range(Periods) # Set of time periods
K = range(Products) # Set of products

''' Simulated historical data generator for quantities, prices and demand of products in each period.
    Returns:
        - M_kt: (dict) subset of suppliers that offer k \in K on t \in T
        - K_it: (dict) subset of products offered by i \in M on t \in T
        - q: (dict) quantity of k \in K offered by supplier i \in M on t \in T
        - p: (dict) price of k \in K offered by supplier i \in M on t \in T
        - dem: (dict) demand of k \in K on t \in T
    Parameters:
        - num_periods: (float) number of historical time periods to simulate, INCLUDING THE REALIZED DEMAND PERIODS
        - lambd: (float) lambda used for demand estimation
        - K: (iter) set of products
        - M: (iter) set of suppliers
        - Periods: (int) number of decision periods of the Rolling Horizon model'''
def gen_quantities_prices_demand(num_periods,lambd,K,M,Periods):
    
    TW = range(0-num_periods,Periods)
    
    M_kt= {}
    # In each time period, for each product
    for k in K:
        for t in TW:
            # Random number of suppliers that offer k in t
            sup = randint(1,len(M))
            M_kt[k,t] = list(M)
            # Random suppliers are removed from subset, regarding {sup}
            for ss in range(len(M)-sup):
                a = int(randint(0,len(M_kt[k,t])-1))
                del M_kt[k,t][a]
    
    # Products offered by each supplier on each time period, based on M_kt
    K_it = {(i,t):[k for k in K if i in M_kt[k,t]] for i in M for t in TW}
    
    # Random quantity of available product k, provided by supplier i on t
    q = {(i,k,t):randint(1,15) if i in M_kt[k,t] else 0 for i in M for k in K for t in TW}
    # Random price of available product k, provided by supplier i on t
    p = {(i,k,t):randint(1,500) if i in M_kt[k,t] else 1000 for i in M for k in K for t in TW}
    
    # Random demand of product k on t
    dem = {(k,t):randint(15,45) for k in K for t in TW}
    
    return M_kt, K_it, q, p, dem

''' Rolling horizon model parameters generation function
    Returns:
        - h: (dict) holding cost of k \in K on t \in T
        - O_k: (dict) maximum days that k \in K can be held in inventory
        - c: (dict) transportation cost between nodes i \in V and j \in V
        - I0: (dict) initial inventory level of k \in K, aged o \in O_k
    Parameters:
        - V: (iter) set of nodes in problem network (suppliers + warehouse)
        - T: (iter) set of decision periods in the rolling horizon model
        - K: (iter) set of products
        - M: (iter) set of suppliers
'''
def gen_params(V,T,K,M):
    # Random holding cost of product k on t
    h = {(k,t):randint(1,500) for k in K for t in T}
    # Fixed backorder cost of product k on t
    g = {(k,t):600 for k in K for t in T}
    
    # '''
    # ######################
    # ######### Delete later
    # ######################
    # '''
    # h = {(k,t):h[k,t]/20 for k in K for t in T}
    # g = {(k,t):g[k,t]/10 for k in K for t in T}
    
    # Maximum days that product k can be held in inventory before rotting
    O_k = {k:randint(1,len(T)) for k in K}
    
    # Suppliers locations in grid
    size_grid = 1000
    coor = {i:(randint(0, size_grid), randint(0, size_grid)) for i in V}
    # Transportation cost between nodes i and j, estimated using euclidean distance
    c = {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in V for j in V if i!=j}
    
    # Initial inventory level of product k, aged o at the beginning of the planning horizon
    I0 = {(k,o):0 for k in K for o in range(1,O_k[k]+1)}
    
    return h, g, O_k, c, I0

''' Creates historical datasets'''
M_kt, K_it, q, p, d = gen_quantities_prices_demand(Historical, lambda1, K, M, Periods)

''' Creates model fixed parameters'''
h, g, O_k, c, I0 = gen_params(V, T, K, M)

I00 = {a:I0[a] for a in I0}


#%% Sample paths Generator

''' Sample value generator function.
    Returns a generated random number using acceptance-rejection method.
    Parameters:
    - hist: (list) historical dataset that is used as an empirical distribution for
            the random number generation'''
def sim(hist):
    Te = len(hist)
    sorted_data = sorted(hist)    
    
    prob, value = [], []
    for t in range(Te):
        prob.append((t+1)/Te)
        value.append(sorted_data[t])
    
    # Generates uniform random value for acceptance-rejection testing
    U = random()
    # Tests if the uniform random falls under the empirical distribution
    test = [i>U for i in prob]    
    # Takes the first accepted value
    sample = value[test.index(True)]
    
    return sample

''' Sample paths generator function.
    Returns:
        - M_kts: (dict) subset of suppliers that offer k \in K on t \in T in sample path s \in Sam
        - K_its: (dict) subset of products offered by i \in M on t \in T in sample path s \in Sam
        - q_s: (dict) quantity of k \in K offered by supplier i \in M on t \in T in sample path s \in Sam
        - p_s: (dict) price of k \in K offered by supplier i \in M on t \in T in sample path s \in Sam
        - dem_s: (dict) demand of k \in K on t \in T in sample path s \in Sam
        - Q_s: (float) feasible vehicle capacity to use in rolling horizon model in sample path s \in Sam
        - F_s: (iter) set of vehicles in sample path s \in Sam
    Parameters:
        - q: (dict) historical data of quantity of k \in K offered by supplier i \in M on t \in T
        - p: (dict) historical data of price of k \in K offered by supplier i \in M on t \in T
        - d: (dict) historical data of demand of k \in K on t \in T
        - M_kt: (dict) historical data of subset of suppliers that offer k \in K on t \in T
        - K_it: (dict) historical data of subset of products offered by i \in M on t \in T
        - horizon: (int) number of periods to simulate in sample paths
        - today: (int) current time period on the Rolling Horizon model
        - K: (iter) set of products
        - M: (iter) set of suppliers
        - Sam: (int) number of sample paths to create
        - today_is_known: (bool) whether today's parameters are known beforehand or not'''
def gen_sim_paths(q, p, d, M_kt, K_it, today, K, M, Sam, horizon, today_is_known):
    
    if today_is_known:
        TW = horizon[1:]
        Hist_T = range(-Historical,today+1)
    else:
        Hist_T = range(-Historical,today)
    
    M_kts,K_its = {},{}
    q_s,p_s,dem_s,Q_s,F_s = {},{},{},{},{}
    
    for s in Sam:
        # For each product, on each period chooses a random subset of suppliers that the product has had
        M_kts[s] = {(k,t):[M_kt[k,tt] for tt in Hist_T][randint(0,len(Hist_T)-1)] for k in K for t in TW}
        
        # Products offered by each supplier on each time period, based on M_kts
        K_its[s] = {(i,t):[k for k in K if i in M_kts[s][k,t]] for i in M for t in TW}
        
        # For each supplier and product, on each period chooses a quantity to offer using the sample value generator function
        q_s[s] = {(i,k,t):sim([q[i,k,tt] for tt in Hist_T if q[i,k,tt] > 0]) if i in M_kts[s][k,t] else 0 for i in M for k in K for t in TW}
        
        # For each supplier and product, on each period chooses a price using the sample value generator function
        p_s[s] = {(i,k,t):sim([p[i,k,tt] for tt in Hist_T if p[i,k,tt]<1000]) if i in M_kts[s][k,t] else 1000 for i in M for k in K for t in TW}
        
        # For each supplier and product, on each period chooses a demand using the sample value generator function
        dem_s[s] = {(k,t):sim([d[k,tt] for tt in Hist_T]) for k in K for t in TW}
        
        # If today's information is known beforehand, today's parameters are taken from the historical datasets
        if today_is_known:
            for k in K:
                M_kts[s][k,0] = M_kt[k,today]
                dem_s[s][k,0] = d[k,today]
                for i in M:
                    q_s[s][i,k,0] = q[i,k,today]
                    p_s[s][i,k,0] = p[i,k,today]
            for i in M:
                K_its[s][i,0] = K_it[i,today]
        
        # Vehicle capacity estimation
        Q_s[s] = 1.2*50
        
        # Set of vehicles, based on estimated required vehicles
        F_s[s] = range(int(sum(dem_s[s].values())/Q_s[s])+1)
        
    return M_kts, K_its, q_s, p_s, dem_s, Q_s, F_s


#%% Models functions

''' Inventory Management LP model
    Returns:
        - ii: (dict) inventory level of product k \in K aged o \in O_k at the end of the current decision period of the Rolling Horizon
        - r: (dict) quantity of product k \in K to be replenished on the current decision period of the Rolling Horizon
        - b: (dict) backorder quantity of product k \in K on the current decision period of the Rolling Horizon
        - I0: (dict) initial inventory level of product k \in K aged o \in O_k for the next decision period of the Rolling Horizon
    Parameters:
        - dec_p: (int) current decision period on the Rolling Horizon model
        - TW: (iter) set of decision periods for the lookahead model
        - d_s: (dict) demand of product k \in K on t \in TW in sample path s \in Sam
        - q_s: (dict) available quantity of product k \in K offered by supplier i \in M on t \in TW in sample path s \in Sam
        - M_kts: (dict) subset of suppliers that offer product k \in K on t \in TW in sample path s \in Sam
        - I0: (dict) initial inventory level of product k \in K aged o \in O_k on the current decision period
'''
def Inventory_LP(dec_p,TW,d_s,q_s,M_kts,I0):
    m = gu.Model('IM_perishable')
    
    ''' Decision Variables '''
    ###### Inventory variables
    # Quantity of product k \in K that is replenished on t \in TW, for sample path s \in Sam
    r = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"r_{k,t,s}") for k in K for t in TW for s in Sam}
    # Quantity of product k \in K aged o \in O_k shipped to final customer on t, for sample path s \in Sam
    y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"y_{k,t,o,s}") for k in K for t in TW for o in range(1, O_k[k]+1) for s in Sam}
    # Backorder quantity of product k \in K on t \in TW, for sample path s \in Sam
    b = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS,name=f"b_{k,t,s}") for k in K for t in TW for s in Sam}
    
    ''' State Variables '''
    # Inventory level of product k \in K aged o \in O_k on t \in TW, for sample path s \in Sam
    ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name=f"i_{k,t,o,s}") for k in K for t in TW for o in range(1, O_k[k]+1) for s in Sam}
    
    ''' Constraints '''
    for s in Sam:
        #Inventory constraints
        ''' For each product k aged o, the inventory level on the first time period is
        the initial inventory level minus what was shipped of it in that same period'''
        for k in K:
            for o in range(1, O_k[k]+1):
                if o > 1:
                    m.addConstr(ii[k,0,o,s] == I0[k,o]-y[k,0,o,s])
        
        ''' For each product k on t, its inventory level aged 0 is what was replenished on t (r)
        minus what was shipped (y)'''
        for k in K:
            for t in TW:
                m.addConstr(ii[k,t,1,s] == r[k,t,s]-y[k,t,1,s])
                
        ''' For each product k, aged o, on t, its inventory level is the inventory level
        aged o-1 on the previous t, minus the amount of it that's shipped '''
        for k in K:
            for t in TW:
                for o in range(1, O_k[k]+1):
                    if t>0 and o>1:
                        m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s]-y[k,t,o,s])                
        
        ''' The amount of product k that is shipped to the final customer on t is equal to its demand'''
        for k in K:
            for t in TW:
                m.addConstr(gu.quicksum(y[k,t,o,s] for o in range(1, O_k[k]+1)) + b[k,t,s] == d_s[s][k,t])
        
        ''' Cannot buy more than the total farmers supply of product k on t '''
        for t in TW:
            for k in K:
                m.addConstr(r[k,t,s] <= sum(q_s[s][i,k,t] for i in M))
        
        ''' Non-Anticipativity Constraints'''
        for k in K:
            m.addConstr(r[k,0,s] == gu.quicksum(r[k,0,s1] for s1 in Sam)/len(Sam))
        
        for k in K:
            m.addConstr(b[k,0,s] == gu.quicksum(b[k,0,s1] for s1 in Sam)/len(Sam))
        
        for k in K:
            for o in range(1, O_k[k]+1):
                m.addConstr(y[k,0,o,s] == gu.quicksum(y[k,0,o,s1] for s1 in Sam)/len(Sam))
    
    ''' Backorders '''
    m.addConstr(gu.quicksum(b[k,t,s] for k in K for t in TW for s in Sam)/len(Sam) <= 0.5*gu.quicksum(d_s[s][k,t] for k in K for t in TW for s in Sam)/len(Sam))
    
    ''' Costs - Objective Function '''
    holding = gu.quicksum(h[k,t+dec_p]*ii[k,t,o,s] for s in Sam for k in K for t in TW for o in range(1, O_k[k]+1))
    backl = gu.quicksum(g[k,t+dec_p]*b[k,t,s] for s in Sam for k in K for t in TW)
    m.setObjective((holding+backl)/len(Sam))
    
    # Runs model
    m.update()
    m.setParam("OutputFlag",0)
    m.optimize()
    
    ''' Initial inventory level updated for next decision period on the Rolling Horizon'''
    I0 = {(k,o):ii[k,0,o-1,0].X if o>1 else 0 for k in K for o in range(1,O_k[k]+1)}
    
    ''' Saves results '''
    ii = {(k,t,o,s):ii[k,t,o,s].X for k in K for o in range(1,O_k[k]+1) for t in TW for s in Sam}
    r = {(k,t,s):round(r[k,t,s].X) for k in K for t in TW for s in Sam}
    b = {(k,t,s):b[k,t,s].X for k in K for t in TW for s in Sam}
    
    return ii, r, b, I0


''' Greedy Purchasing algorithm
    Returns:
        - z: (dict) quantity of product k \in K to buy from supplier i \in M
    Parameters:
        - r: (dict) (dict) quantity of product k \in K to be replenished on the current decision period of the Rolling Horizon
'''
def Purchasing_greedy(r,TW,p_s,q_s):
    z = {(i,k,t,s):0. for i in M for k in K for t in TW for s in Sam}
    for s in Sam:
        for t in TW:
            
            for k in K:
                
                # List of prices offered by suppliers for product k on the current decision period
                prices = [p_s[s][i,k,t] for i in M]
                
                # As long as I still haven't bought everything I need of product k
                while r[k,t,s] > 0:
                    # Get supplier that offers lowest price. Adds +1 bc suppliers start at 1 in M
                    i = prices.index(min(prices))+1
                    # Discards the selected supplier to account for possible ties in suppliers' prices
                    prices[i-1] = 1e3
                    # Decides quantity to buy from the selected supplier
                    z[i,k,t,s] = min(q_s[s][i,k,t],r[k,t,s])
                    # Updates quantity left to buy
                    r[k,t,s] -= z[i,k,t,s]
    
    return z


''' Routing tour generator
    Returns:
        - x: (dict) whether supplier j \in V is visited after supplier i \in V on t \in TW on sample path s \in S
    Parameters:
        - TW: (iter) set of decision periods of the lookahead model on the current decision period of the Rolling Horizon model
'''
def Routing_random(TW):
    ''' Tour creating function '''
    def create_tour():
        M1 = list(M).copy()
        M1 = list(M).copy()
        tour = []
        while M1:
            i = randint(0,len(M1)-1)
            tour.append(M1[i])
            M1.pop(i)
        return tour
    
    ''' Routing decisions setting function '''
    def routing_decisions(x,tour):
        for i in range(len(tour)-1):
            x[tour[i],tour[i+1],t,s] = 1
        x[0,tour[0],t,s] = 1
        x[tour[-1],0,t,s] = 1
        
        return x
    
    x = {(i,j,t,s):0 for i in V for j in V for t in TW for s in Sam if i != j}
    for t in TW:
        if t == 0:
            tour = create_tour()
            for s in Sam:
                x = routing_decisions(x, tour)
        else:
            for s in Sam:
                tour = create_tour()
                x = routing_decisions(x,tour)
    
    return x



#%% Models Implementation

# Lookahead periods on each sample path (including today)
horizon_T = 5
samples = 3
Sam = range(samples)

res = {}
''' Rolling '''
for tau in T:
    
    # Time window adjustment
    adj_horizon = min(horizon_T,Periods-tau)
    TT = range(adj_horizon)
    
    ''' Sample paths generation '''
    M_kts, K_its, q_s, p_s, d_s, Q_s, F_s = gen_sim_paths(q, p, d, M_kt, K_it, tau,
                                                          K, M, Sam, TT, True)
    
    ''' Inventory Management Decisions '''
    ii,r,b,I0 = Inventory_LP(tau, TT, d_s, q_s, M_kts, I0)
    
    ''' Purchasing Decisions '''
    z = Purchasing_greedy(r,TT,p_s,q_s)
    
    ''' Routing Decisions '''
    x = Routing_random(TT)
    
    res[tau] = (ii,b,z,TT,d_s,p_s,q_s,x)


#%% Results translation

ii = {tau:res[tau][0] for tau in T}
b = {tau:res[tau][1] for tau in T}
z = {tau:res[tau][2] for tau in T}
TW = {tau:res[tau][3] for tau in T}
d_s = {tau:res[tau][4] for tau in T}
p_s = {tau:res[tau][5] for tau in T}
q_s = {tau:res[tau][6] for tau in T}
x = {tau:res[tau][7] for tau in T}

I0 = {}
for tau in T:
    I0[tau] = {}
    
    for s in Sam:
        for k in K:
            for o in range(1,O_k[k]+1):
                
                # o = 1
                if o == 1:
                    for t in TW[tau]:
                        I0[tau][k,t,o,s] = 0
                else:
                    
                    # t = 0, o > 1
                    if tau > 0:
                        I0[tau][k,0,o,s] = ii[tau-1][k,0,o-1,s]
                    else:
                        I0[tau][k,0,o,s] = I00[k,o]
                    
                    # t > 0, o > 1
                    for t in TW[tau][1:]:
                        I0[tau][k,t,o,s] = ii[tau][k,t-1,o-1,s]
            
#%% Visualization    
            

# Max value for first chart axis
initi = {(s,tau,t):sum(I0[tau][k,t,o,s] for k in K for o in range(1,O_k[k]+1)) for s in Sam for tau in T for t in TW[tau]}
repl = {(s,tau,t):sum(z[tau][i,k,t,s] for i in M for k in K) for s in Sam for tau in T for t in TW[tau]}
ub1 = max([initi[s,tau,t]+repl[s,tau,t] for s in Sam for tau in T for t in TW[tau]])
ub1 = (int(ub1/50)+2)*50

# Max value for second chart axis
purch = {(s,tau,t):sum(p_s[tau][s][i,k,t]*z[tau][i,k,t,s] for i in M for k in K) for s in Sam for tau in T for t in TW[tau]}
hold = {(s,tau,t):sum(h[k,tau+t]*ii[tau][k,t,o,s] for k in K for o in range(1,O_k[k]+1)) for s in Sam for tau in T for t in TW[tau]}
backo = {(s,tau,t):sum(g[k,tau+t]*b[tau][k,t,s] for k in K) for s in Sam for tau in T for t in TW[tau]}
rout = {(s,tau,t):sum(c[i,j]*x[tau][i,j,t,s] for i in V for j in V if i !=j) for s in Sam for tau in T for t in TW[tau]}
ub2 = max([purch[s,tau,t]+hold[s,tau,t]+backo[s,tau,t]+rout[s,tau,t] for s in Sam for tau in T for t in TW[tau]])
ub2 = (int(ub2/2e4)+1)*2e4

cols = {0:"navy",1:"deeppink",2:"chocolate",3:"seagreen", 4:"dimgrey", 5:"teal", 6:"olive", 7:"darkmagenta"}

for tau in T:
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(13,10))
    
    ''' First chart '''
    for s in Sam:
        ax1.plot([t for t in range(tau,tau+len(TW[tau]))],[sum(d_s[tau][s][k,t] for k in K) for t in TW[tau]],color=cols[s])
    
    for t in range(tau+1):
        initi = sum(I0[t][k,0,o,0] for k in K for o in range(1,O_k[k]+1))
        repl = sum(z[t][i,k,0,0] for i in M for k in K)
        backo = sum(b[t][k,0,0] for k in K)
        ax1.bar(x=t-0.2, height=initi, color="darkgoldenrod",width=0.4)
        ax1.bar(x=t-0.2, height=repl, bottom=initi, color="indigo",width=0.4)
        ax1.bar(x=t+0.2, height=backo, color="forestgreen", width=0.4)
    for t in range(1,len(TW[tau])):
        xx = tau+t
        initi = [sum(I0[tau][k,t,o,s] for k in K for o in range(1,O_k[k]+1)) for s in Sam]
        repl = [sum(z[tau][i,k,t,s] for i in M for k in K) for s in Sam]
        tc = [initi[s]+repl[s] for s in Sam]
        backo = [sum(b[tau][k,t,s] for k in K) for s in Sam]
        ax1.bar(x=xx-0.2, height=sum(initi)/len(Sam), color="khaki", width=0.4)
        ax1.bar(x=xx-0.2, height=sum(repl)/len(Sam), bottom=sum(initi)/len(Sam), color="mediumpurple", width=0.4)
        ax1.axvline(x=xx-0.2,ymin=min(tc)/ub1,ymax=max(tc)/ub1,color="black",marker="_",mew=1.5,ms=8)
        ax1.bar(x=xx+0.2, height=sum(backo)/len(Sam), color="palegreen", width=0.4)
        ax1.axvline(x=xx+0.2,ymin=min(backo)/ub1,ymax=max(backo)/ub1,color="black",marker="_",mew=1.5,ms=8)
    ax1.plot([tt for tt in range(tau+1)],[sum(d[k,0+tt] for k in K) for tt in range(tau+1)],linestyle="-",marker="*",markersize=12,color="black")
    ax1.set_xlim(-0.5,Periods-0.5)
    ax1.set_ylim(0,ub1)
    ax1.set_xlabel("Time period")
    ax1.set_ylabel("Quantity")
    ax1.bar(x=tau,height=0,color="forestgreen",label="Backorders")
    ax1.bar(x=tau,height=0,color="indigo",label="Replenishment")
    ax1.bar(x=tau,height=0,color="darkgoldenrod",label="Initial Inv. Level")
    ax1.plot(tau,0,color="black",linestyle="-",marker="*",markersize=9,label="Demand")
    ax1.legend(loc="upper right",ncol=2)
    
    ''' Second chart ''' 
    for t in range(tau+1):
        purch = sum(p_s[t][0][i,k,0]*z[t][i,k,0,0] for i in M for k in K)
        hold = sum(h[k,t]*ii[t][k,0,o,0] for k in K for o in range(1,O_k[k]+1))
        backo = sum(g[k,t]*b[t][k,0,0] for k in K)
        rout = sum(c[i,j]*x[t][i,j,0,0] for i in V for j in V if i != j)
        ax2.bar(x=t, height=purch, color="indigo")
        ax2.bar(x=t, height=hold, bottom=purch, color="darkgoldenrod")
        ax2.bar(x=t, height=backo, bottom=purch+hold, color="forestgreen")
        ax2.bar(x=t, height=rout, bottom=purch+hold+backo, color="darkmagenta")
    
    for t in range(1,len(TW[tau])):
        xx = tau+t
        purch = [sum(p_s[tau][s][i,k,t]*z[tau][i,k,t,s] for i in M for k in K) for s in Sam]
        hold = [sum(h[k,xx]*ii[tau][k,t,o,s] for k in K for o in range(1,O_k[k]+1)) for s in Sam]
        backo = [sum(g[k,xx]*b[tau][k,t,s] for k in K) for s in Sam]
        rout = [sum(c[i,j]*x[tau][i,j,t,s] for i in V for j in V if i != j) for s in Sam]
        tc = [purch[s]+hold[s]+backo[s]+rout[s] for s in Sam]
        ax2.bar(x=xx, height=sum(purch)/len(Sam), color="mediumpurple")
        ax2.bar(x=xx, height=sum(hold)/len(Sam), bottom=sum(purch)/len(Sam), color="khaki")
        ax2.bar(x=xx, height=sum(backo)/len(Sam), bottom=(sum(purch)+sum(hold))/len(Sam), color="palegreen")
        ax2.bar(x=xx, height=sum(rout)/len(Sam), bottom=(sum(purch)+sum(hold)+sum(backo))/len(Sam), color="violet")
        ax2.axvline(x=xx,ymin=min(tc)/ub2,ymax=max(tc)/ub2,color="black",marker="_",mew=1.5,ms=8)
    ax2.set_xlim(0-0.5,Periods-0.5)
    ax2.set_ylim(0,ub2)
    ax2.bar(x=tau,height=0,color="darkmagenta",label="Routing")
    ax2.bar(x=tau,height=0,color="forestgreen",label="Backorders")
    ax2.bar(x=tau,height=0,color="darkgoldenrod",label="Holding")
    ax2.bar(x=tau,height=0,color="indigo",label="Purchase")
    ax2.legend(loc="upper right")
    ticks = [i for i in range(0,int(ub2+ub2/10),int(ub2/10))]
    ax2.set_yticks(ticks=ticks)
    ax2.set_yticklabels(["${:,.0f}K".format(int(i/1e3)) for i in ticks])
    ax2.set_ylabel("Total Cost")
    ax2.set_xlabel("Time period")

#%% Checking



#%%

for tau in T:
    print(f"\nDay {tau}:")
    for k in K:
        a = [i for i in M if z[tau][i,k] > 0]
        print(f"Product {k}: {a}, bought {r[tau][k]}")



