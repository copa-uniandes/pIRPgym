################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time; from termcolor import colored
from random import random, seed, randint, shuffle
import networkx as nx

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

### Gym & OR-Gym
import gym; #from gym import spaces
from or_gym import utils

class policies():

    def __init__(self) -> None:
        
        pass

    def Genera_ruta_at_t(self, solucionTTP, t, max_cij, c, Q):

        Info_Route, solucionTTP = self.Crea_grafo_aumentado_at_t(t, solucionTTP, max_cij, c)
        FO_Routing, Rutas_finales = self.Genera_Rutas_CVRP_at_t(Info_Route, solucionTTP, t, c, Q)

        return Rutas_finales, solucionTTP, FO_Routing


    def det_rolling_horizon(self,state, _, env):

        solucionTTP = {0:[  np.zeros(env.M+1, dtype=bool), 
                            np.zeros(env.M+1, dtype=int), 
                            np.zeros((env.M+1, env.K), dtype=bool), 
                            np.zeros((env.M+1, env.K), dtype=int), 
                            np.full(env.M+1, -1, dtype = int), 
                            np.zeros(env.M+1, dtype=int), 
                            np.zeros(env.K, dtype=int), 0, 0]}

        # State 
        I_0 = state.copy()
        sample_paths = _['sample_paths']

        # Look ahead window     
        Num_periods = _['sample_path_window_size']
        T = range(Num_periods)

        # Iterables
        M = env.Suppliers;  K = env.Products

        # Initialization routing cost
        C_MIP = {(i,t):env.c[0,i]+env.c[i,0] for t in T for i in env.Suppliers} 

        m = gu.Model('Inventory')

        # Variables    
        # How much to buy from supplier i of product k at time t 
        z = {(i,k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t))) for t in T for k in K for i in sample_paths[('M_k',0)][(k,t)]}

        # 1 if supplier i is selected at time t, 0 otherwise
        w = {(i,t):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t))) for t in T for i in M}

        # Final inventory of product k of old o at time t 
        ii = {(k,t,o):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,o))) for k in K for t in T for o in range(env.O_k[k] + 1)}

        # Units sold of product k at time t of old age o
        y = {(k,t,o):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_"+str((k,t,o))) for k in K for t in T for o in range(env.O_k[k] + 1)}

        # Units in backorders of product k at time t
        bo = {(k,t):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t))) for t in T for k in K}

        #Inventory constrains
        for k in K:
            for t in T:
                m.addConstr(ii[k,t,0] == gu.quicksum(z[i,k,t] for i in sample_paths[('M_k',0)][(k,t)]) - y[k,t,0])
                
        for k in K:
            for o in env.Ages[k]:
                m.addConstr(ii[k,0,o] == I_0[k,o] - y[k,0,o])
                
        for k in K:
            for t in T:
                for o in env.Ages[k]:
                    if t > 0:
                        m.addConstr(ii[k,t,o] == ii[k,t-1,o-1] - y[k,t,o])

        for k in K: 
            for t in T:
                m.addConstr(gu.quicksum(y[k,t,o] for o in range(env.O_k[k] + 1)) + bo[k,t] == sample_paths[('d',0)][k,t])   


        #Purchase constrains
        for t in T:
            for k in K:
                for i in sample_paths[('M_k',0)][k,t]: 
                    m.addConstr(z[i,k,t] <= env.sample_paths[('q',0)][i,k,t]*w[i,t])
                    
        for t in T:
            for i in env.Suppliers:
                m.addConstr(gu.quicksum( z[i,k,t] for k in K if (i,k,t) in z) <= env.Q)
        
        compra = gu.quicksum(sample_paths['p',0][i,k,t]*z[i,k,t] for k in K for t in T for i in sample_paths[('M_k',0)][k,t] ) + \
            gu.quicksum(env.back_o_cost*bo[k,t] for k in K for t in T)
        
        #for the following H periods I will work with Expected Value
        ruta = gu.quicksum(C_MIP[i,t]*w[i,t] for i in M for t in T) 
            
        m.setObjective(compra+ruta)
                
        m.update()
        m.setParam('OutputFlag',0)
        m.optimize()

        # Purchase
        purchase = {(i,k): 0 for i in M for k in K}

        for k in K:
            for i in sample_paths[('M_k',0)][k,0]:
                purchase[i,k] = z[i,k,0].x
                if purchase[i,k]>0:
                    solucionTTP[0][0][i] = True
                    solucionTTP[0][1][i]+= purchase[i,k]
                    solucionTTP[0][2][i][k]=True
                    solucionTTP[0][3][i][k]=purchase[i,k]
                    solucionTTP[0][6][k]+=purchase[i,k]

        # Demand compliance
        demand_compliance = {(k,o):y[k,0,o].x for k in K for o in range(env.O_k[k] + 1)}

        # Back-orders
        double_check = {(k,t): bo[k,0].x for k in K}
        
        #Updated inventory for next period t
        I_1 = {}
        for k in env.Products:
            for o in range(env.O_k[k] + 1):
                I_1[k,o] = ii[k,0,o].x

        Rutas_finales, solucionTTP, solucionTTP[0][8]  = self.Genera_ruta_at_t(solucionTTP, 0, max(env.c.values())*2, env.c, env.Q)

        #print(Rutas_finales)
        rutas = []
        for key in Rutas_finales[0].keys():
            rutas.append(Rutas_finales[0][key][0])

        return [rutas, purchase, demand_compliance]#, double_check, I_1


    def stochastic_rolling_horizon(self, state, _, env):
    
        solucionTTP = {0:[  np.zeros(env.M+1, dtype=bool), 
                                np.zeros(env.M+1, dtype=int), 
                                np.zeros((env.M+1, env.K), dtype=bool), 
                                np.zeros((env.M+1, env.K), dtype=int), 
                                np.full(env.M+1, -1, dtype = int), 
                                np.zeros(env.M+1, dtype=int), 
                                np.zeros(env.K, dtype=int), 0, 0]}

        # State
        I_0 = state.copy()
        sample_paths = _['sample_paths']

        # Look ahead window     
        Num_periods = _['sample_path_window_size']
        T = range(Num_periods)

        # Iterables
        M = env.Suppliers; K = env.Products; S = env.Samples

        # Initialization routing cost
        C_MIP = {(i,t):env.c[0,i]+env.c[i,0] for t in T for i in env.Suppliers} 

        m = gu.Model('Inventory')

        # Variables    
        # How much to buy from supplier i of product k at time t 
        z = {(i,k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="z_"+str((i,k,t,s))) for t in T for k in K for s in S for i in env.M_kt[k,env.t + t]}
        tuples = [(i,k,t,s) for t in T for k in K for s in S for i in env.M_kt[k,env.t + t]]

        # 1 if supplier i is selected at time t, 0 otherwise
        w = {(i,t,s):m.addVar(vtype=gu.GRB.BINARY, name="w_"+str((i,t,s))) for t in T for i in M for s in S}

        # Final inventory of product k of old o at time t 
        ii = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="i_"+str((k,t,o,s))) for k in K for t in T for o in range(env.O_k[k] + 1) for s in S}

        # Units sold of product k at time t of old age o
        y = {(k,t,o,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="y_"+str((k,t,o,s))) for k in K for t in T for o in range(env.O_k[k] + 1) for s in S}

        # Units in backorders of product k at time t
        bo = {(k,t,s):m.addVar(vtype=gu.GRB.CONTINUOUS, name="bo_"+str((k,t,s))) for t in T for k in K for s in S}

        for s in S:
            ''' Inventory constraints '''
            for k in K:
                for t in T:
                    m.addConstr(ii[k,t,0,s] == gu.quicksum(z[i,k,t,s] for i in env.M_kt[(k,env.t + t)]) - y[k,t,0,s])
                    
            for k in K:
                for o in env.Ages[k]:
                    m.addConstr(ii[k,0,o,s] == I_0[k,o] - y[k,0,o,s])
                    
            for k in K:
                for t in T:
                    for o in env.Ages[k]:
                        if t > 0:
                            m.addConstr(ii[k,t,o,s] == ii[k,t-1,o-1,s] - y[k,t,o,s])

            for k in K: 
                for t in T:
                    m.addConstr(gu.quicksum(y[k,t,o,s] for o in range(env.O_k[k] + 1)) + bo[k,t,s] == sample_paths['d'][t,s][k])   


            ''' Purchase constraints '''
            for t in T:
                for k in K:
                    for i in env.M_kt[k,env.t + t]: 
                        m.addConstr(z[i,k,t,s] <= sample_paths['q'][t,s][i,k]*w[i,t,s])
                        
            for t in T:
                for i in M:
                    m.addConstr(gu.quicksum( z[i,k,t,s] for k in K if (i,k,t,s) in z) <= env.Q)
        
            '''' NON-ANTICIPATIVITY CONSTRAINTS '''
            for k in K:

                for i in env.M_kt[k,env.t]:
                    m.addConstr(z[i,k,0,s] == gu.quicksum(z[i,k,0,ss] for ss in S)/len(S))
                
                for o in range(env.O_k[k] + 1):
                    m.addConstr(y[k,0,o,s] == gu.quicksum(y[k,0,o,ss] for ss in S)/len(S))
            
            for i in M:
                m.addConstr(w[i,0,s] == gu.quicksum(w[i,0,ss] for ss in S)/len(S))

        compra = gu.quicksum(env.p_t[env.t][i,k]*z[i,k,t,s] for k in K for t in T for s in S for i in env.M_kt[k,env.t + t])/len(S) + \
            env.back_o_cost*gu.quicksum(bo[k,t,s] for k in K for t in T for s in S)/len(S)
        
        ruta = gu.quicksum(C_MIP[i,t]*w[i,t,s] for i in M for t in T for s in S)
        
        m.setObjective(compra+ruta)
                
        m.update()
        m.setParam('OutputFlag',0)
        m.optimize()
        print(m.Status)

        # Purchase
        purchase = {(i,k): 0 for i in M for k in K}

        for k in K:
            for i in env.M_kt[k,env.t]:
                purchase[i,k] = z[i,k,0,0].x
                if purchase[i,k]>0:
                    solucionTTP[0][0][i] = True
                    solucionTTP[0][1][i]+= purchase[i,k]
                    solucionTTP[0][2][i][k]=True
                    solucionTTP[0][3][i][k]=purchase[i,k]
                    solucionTTP[0][6][k]+=purchase[i,k]
        
        # Demand compliance
        demand_compliance = {(k,o):y[k,0,o,0].x for k in K for o in range(env.O_k[k] + 1)}

        # Back-orders
        double_check = {(k,t): bo[k,0,0].x for k in K}
        
        Rutas_finales, solucionTTP, solucionTTP[0][8]  = self.Genera_ruta_at_t(solucionTTP, 0, max(env.c.values())*2, env.c, env.Q)

        rutas = []
        for key in Rutas_finales[0].keys():
            rutas.append(Rutas_finales[0][key][0])

        action = [rutas, purchase, demand_compliance]
        
        I0 = {}
        for t in T: 
            I0[t] = {}
            for s in S:  
                I0[t][s] = {}
                for k in K:
                    for o in env.Ages[k]:
                        I0[t][s][k,o] = ii[k,t,o,s].x
        zz = {}
        for t in T:
            zz[t] = {}
            for s in S:
                zz[t][s] = {}
                for k in K:
                    for i in env.M_kt[k,env.t + t]:
                        zz[t][s][i,k] = z[i,k,t,s].x 
        bb = {}
        for t in T:
            bb[t] = {}
            for s in S:
                bb[t][s] = {}
                for k in K:
                    bb[t][s][k] = bo[k,t,s].x
        la_decisions = [I0, zz, bb]

        return action, la_decisions


    def Crea_grafo_aumentado_at_t(self, t, solucionTTP, max_cij, c):
        Info_Route = {}
        nodes = [i for i, x in enumerate(solucionTTP[t][0]) if x] #See which suppliers are in the solution
        nodes.insert(0,0)

        if len(nodes)>1: 
            Info_Route[t] = [nodes, len(nodes)]
            
            matrix_cij = np.full((Info_Route[t][1],Info_Route[t][1]), max_cij) 
            
            #Build distance matrix just for supplier selected
            for i in range(len(Info_Route[t][0])):
                for j in range(len(Info_Route[t][0])):
                    if i != j :
                        matrix_cij[i,j] = c[Info_Route[t][0][i],Info_Route[t][0][j]]
                        
            #Run nearest neighbor algorithm
            Ruta = []            
            Cabeza = 0 
            Ruta.append(Cabeza)
            
            while len(Ruta) < Info_Route[t][1]:
                minimo = max_cij
                for i in range(len(nodes)):
                    if matrix_cij[Cabeza][i] < minimo:
                        minimo = matrix_cij[Cabeza][i]
                        Cola = i
                    
                    matrix_cij[i][Cabeza]=max_cij
                    
                Ruta.append(Info_Route[t][0][Cola])
                Cabeza = Cola
                
            Info_Route[t].append(Ruta) #General route

        return Info_Route, solucionTTP


    #Build Augmented graph following general tour
    def Genera_Rutas_CVRP_at_t(self, Info_Route, solucionTTP, t, c, Q):
        Rutas_finales = {}
        FO_Rutas = {}
        Rutas_finales[t] = {}
        info = {}

        if len(Info_Route) > 0: 
            g = Graph(Info_Route[t][1]) #Define the number of suppliers
            
            #Build the arcs just if they respect vehicle capacity following the general route found
            for i in range(Info_Route[t][1]-1):
                for j in range(i+1, Info_Route[t][1]): 
                    info_ruta = {}   
                    
                    if j - i == 1:
                        camino = [0, Info_Route[t][2][j] ,0]
                        costo = sum(c[camino[k], camino[k+1]] for k in range(len(camino)-1))
                        cap = solucionTTP[t][1][camino[1]]
                        info_ruta[camino[1]] = 1
                
                        if cap <= Q:
                            info[(i,j)] = [camino, cap, costo, info_ruta]
                            g.addEdge(i,j,costo)
                            
                    elif j - i >1:
                        if i ==0:
                            camino=Info_Route[t][2][i:j+1]
                            camino.append(0)
                        else:
                            camino=Info_Route[t][2][i+1:j+1]
                            camino.append(0)
                            camino.insert(0,0)
                        
                        for k in range(1,len(camino)-1):
                            info_ruta[camino[k]] = k
                    
                        costo = sum(c[camino[k], camino[k+1]] for k in range(len(camino)-1))
                        cap = sum(solucionTTP[t][1][camino[k]]  for k in range(1, len(camino)))
                        
                        if cap <=Q:
                            info[(i,j)] = [camino, cap, costo, info_ruta]
                            g.addEdge(i,j,costo)
            
            FO_Routing, Arcos_agregados = g.BellmanFord(0) #Run BellamnFord Algorithm 
            
            #Translate arc in routing solution
            for i in range(len(Arcos_agregados)):
                Rutas_finales[t][i] = info[Arcos_agregados[i]]
                for k in info[Arcos_agregados[i]][3]:
                    solucionTTP[t][4][k]=i
                    solucionTTP[t][5][k] = info[Arcos_agregados[i]][3][k]
                    
        else:
            FO_Routing = 0
                
        return FO_Routing, Rutas_finales

    
#######################    ADDITIONAL ALGORITHMS FOR DETERMINISTIC AND ROLLING HORIZON     #######################
class Graph: # Class to represent a graph

    def __init__(self, vertices):
        self.V = vertices # No. of vertices
        self.graph = []

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
            
    # utility function used to print the solution
    def printArr(self, dist, etiqueta):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}\t\t{2}".format(i, dist[i], etiqueta[i]))
        
    # The main function that finds shortest distances from src to
    # all other vertices using Bellman-Ford algorithm. The function
    # also detects negative weight cycle
    def BellmanFord(self, src):

        # Step 1: Initialize distances from src to all other vertices
        # as INFINITE
        dist = [float("Inf")] * self.V
        etiqueta = [0]*self.V
        dist[src] = 0
        
        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        # path from src to any other vertex can have at-most |V| - 1
        # edges
        for _ in range(self.V - 1):
            # Update dist value and parent index of the adjacent vertices of
            # the picked vertex. Consider only those vertices which are still in
            # queue
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        etiqueta[v] = u
                        
        # print all distance
        #self.printArr(dist, etiqueta)
        
        Cabeza = -1
        Cola = self.V-1
        FO = dist[Cola]
        arcos = []
        while Cabeza !=0 :
            Cabeza = etiqueta[Cola]
            arcos.append((Cabeza,Cola))
            Cola = Cabeza
        
        #print(FO)
        #print(arcos)
        return FO, arcos
        




