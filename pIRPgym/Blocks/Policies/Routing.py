import pandas as np
import numpy as np
from numpy.random import seed,random,choice,randint
from time import time,process_time
from copy import deepcopy
from multiprocess import pool,freeze_support

import gurobipy as gu
#import hygese as hgs
from traitlets import Float

from ..InstanceGenerator import instance_generator
from ..BuildingBlocks import Inventory_management, Routing_management


class Routing():
        options = ['NN, RCL, HGA, CG']

        ''' Nearest Neighbor (NN) heuristic '''
        # Generate routes
        @staticmethod
        def NearestNeighbor(purchase:dict,inst_gen:instance_generator,t:int,price_routes:bool=False) -> tuple:
            """
            Generates a solution using the Nearest Neighbor heuristic for the dCVRP.

            Parameters:
            - purchase (dict): A dictionary representing the purchase order.
            - inst_gen (instance_generator): An instance generator object.
            - t (int): Time parameter.
            - price_routes (bool): If True, calculates and returns reduced costs for routes.

            Returns:
            tuple: A tuple containing routes, total objective function value, route details, and execution time.
                If price_routes is True, also includes reduced costs for each route.

            Example:
            ```python
            routes, FO, details, execution_time = Routing.NearestNeighbor(purchase, inst_gen, t)
            ```
            """
            start = process_time()
            pending_sup,requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            routes = list()
            loads = list()
            FO:float = 0
            distances = list()

            if price_routes:
                reduced_costs = list()

            while len(pending_sup) > 0:
                node:int = 0
                load:int = 0
                route:list = [node]
                distance:float = 0
                while load < inst_gen.Q:
                    target = Routing.Nearest_Neighbor.find_nearest_feasible_node(node,load,distance,pending_sup,requirements,inst_gen)
                    if target == False:
                        break
                    else:
                        load += requirements[target]
                        distance += inst_gen.c[node,target]
                        node = target
                        route.append(node)
                        pending_sup.remove(node)
                route += [0]

                if price_routes:
                    reduced_cost = Routing.PriceRoute(inst_gen,route,purchase,t,routes)
                    reduced_costs.append(reduced_cost)

                routes.append(route)
                distance += inst_gen.c[node,0]
                loads.append(load)
                FO += distance
                distances.append(distance)

            if not price_routes:
                return routes,FO,(distances,loads),process_time()-start
            else:
                return routes,FO,(distances,loads),process_time()-start,reduced_costs

        class Nearest_Neighbor():      
            # Find nearest feasible (by capacity) node
            @staticmethod
            def find_nearest_feasible_node(node:int,load:float,distance:float,pending_sup:list,requirements:dict,inst_gen:instance_generator):
                """
                Finds the nearest feasible node (by capacity and distance) for the Nearest Neighbor heuristic.

                Parameters:
                - node (int): Current node.
                - load (float): Current load.
                - distance (float): Current distance.
                - pending_sup (list): List of pending suppliers.
                - requirements (dict): Dictionary of supplier requirements.
                - inst_gen (instance_generator): An instance generator object.

                Returns:
                int or False: The nearest feasible node if found, False otherwise.

                Example:
                ```python
                target = Routing.Nearest_Neighbor.find_nearest_feasible_node(node, load, distance, pending_sup,
                                                                            requirements, inst_gen)
                ```
                """
                target, dist = False, 1e6
                for candidate in pending_sup:
                    if (inst_gen.c[node,candidate] < dist) and (load + requirements[candidate] <= inst_gen.Q) \
                        and (distance + inst_gen.c[node,candidate] + inst_gen.c[candidate,0] <= inst_gen.d_max):
                        target = candidate
                        dist = inst_gen.c[node,target]
                
                return target
            
        
        @staticmethod
        def RCL_Heuristic(purchase:dict,inst_gen:instance_generator,t,RCL_alphas:list=[0.25],adaptative=True,rd_seed=None,
                          price_routes:bool=False,time_limit:float=15):
            start = process_time()
            _,requirements = Routing.consolidate_purchase(purchase, inst_gen,t)

            best_sol = list()
            best_obj = 1e9
            best_info = ()
            best_time = float

            alpha_performance:dict = {alpha:1/len(RCL_alphas) for alpha in RCL_alphas}

            while process_time()-start < time_limit:
                # Choosing alpha
                RCL_alpha = choice(RCL_alphas,p=[alpha_performance[alpha]/sum(alpha_performance.values()) for alpha in RCL_alphas])    

                if not price_routes:
                    routes,FO,(distances,loads),_ = Routing.RCL_Solution(purchase,inst_gen,t,
                                                                         RCL_alpha,rd_seed,price_routes)

                # if not price_routes:
                #     routes,FO,(distances,loads),_ = Routing.RCL_Solution(purchase,inst_gen,t,
                #                                                          alpha_performance,rd_seed,price_routes)
                # else:
                #     routes,FO,(distances,loads),_,reduced_costs = Routing.RCL_Solution(purchase,inst_gen,t,
                                                                                    #    alpha_performance,rd_seed,price_routes)

                if FO < best_obj:
                    best_sol = routes
                    best_obj = FO
                    best_info = (distances,loads)
                    best_time = process_time()-start

                    if price_routes:    
                        best_r = reduced_costs
                
                # alpha update
                alpha_performance[RCL_alpha] += 1/FO


            if not price_routes: 
                return best_sol,best_obj,best_info,best_time
            else:
                return best_sol,best_obj,best_info,best_time,best_r


        ''' RCL based constructive '''
        @staticmethod
        def RCL_Solution(purchase:dict,inst_gen:instance_generator,t,RCL_alpha:float=0.35,rd_seed=None,price_routes:bool=False) -> tuple:
        # def RCL_Solution(purchase:dict,inst_gen:instance_generator,t,alpha_performance:dict,rd_seed=None,price_routes:bool=False) -> tuple:
            """
            Generates a solution using the RCL (Restricted Candidate List) based constructive heuristic for the dCVRP.

            Parameters:
            - purchase (dict): A dictionary representing the purchase order.
            - inst_gen (instance_generator): An instance generator object.
            - t (int): Time parameter.
            - RCL_alpha (float): Parameter controlling the size of the restricted candidate list.
            - s: Seed for random number generation.
            - price_routes (bool): If True, calculates and returns reduced costs for routes.

            Returns:
            tuple: A tuple containing routes, total objective function value, route details, and execution time.
                If price_routes is True, also includes reduced costs for each route.

            Example:
            ```python
            routes, FO, details, execution_time = Routing.RCL_Heuristic(purchase, inst_gen, t, RCL_alpha=0.35, s=42)
            ```
            """
            start = process_time()
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            if seed != None:
                seed(rd_seed)

            routes = list()
            FO:float = 0
            distances = list()
            loads = list()

            if price_routes:
                reduced_costs = list()

            while len(pending_sup) > 0:
                route,distance,load,pending_sup = Routing.RCL_constructive.generate_RCL_route(RCL_alpha,pending_sup,requirements,inst_gen)

                if price_routes:
                    reduced_cost = Routing.PriceRoute(inst_gen,route,purchase,t,routes)
                    reduced_costs.append(reduced_cost)

                routes.append(route)
                FO += distance
                distances.append(distance)
                loads.append(load)
            
            if not price_routes:
                return routes,FO,(distances,loads),process_time()-start
            else:
                return routes,FO,(distances,loads),process_time()-start,reduced_costs
                 
        class RCL_constructive():
            # Generate candidate from RCL
            @staticmethod
            def generate_RCL_candidate(RCL_alpha,node,load,distance,pending_sup,requirements,inst_gen):
                """
                Generates a candidate from the Restricted Candidate List (RCL).

                Parameters:
                - RCL_alpha (float): Parameter controlling the size of the restricted candidate list.
                - node (int): Current node.
                - load (float): Current load.
                - distance (float): Current distance.
                - pending_sup (list): List of pending suppliers.
                - requirements (dict): Dictionary of supplier requirements.
                - inst_gen (instance_generator): An instance generator object.

                Returns:
                int or False: The selected candidate if found, False otherwise.

                Example:
                ```python
                target = Routing.RCL_constructive.generate_RCL_candidate(0.35, node, load, distance, pending_sup,
                                                                        requirements, inst_gen)
                ```
                """
                feasible_candidates:list = list()
                max_crit:float = -1e9
                min_crit:float = 1e9

                for candidate in pending_sup:
                    d = inst_gen.c[node,candidate] 

                    if distance + d + inst_gen.c[candidate,0] <= inst_gen.d_max and load + requirements[candidate] <= inst_gen.Q:
                        feasible_candidates.append(candidate)
                        max_crit = max(d, max_crit)
                        min_crit = min(d, min_crit)

                upper_bound:float = min_crit + RCL_alpha * (max_crit - min_crit)
                feasible_candidates:list = [i for i in feasible_candidates if inst_gen.c[node, i] <= upper_bound]
                if len(feasible_candidates) != 0:
                    target = choice(feasible_candidates)
                    return target
                else:
                    return False


            # Generate route from RCL
            @staticmethod
            def generate_RCL_route(RCL_alpha,pending_sup,requirements,inst_gen):
                """
                Generates a route using the RCL (Restricted Candidate List) based constructive heuristic.

                Parameters:
                - RCL_alpha (float): Parameter controlling the size of the restricted candidate list.
                - pending_sup (list): List of pending suppliers.
                - requirements (dict): Dictionary of supplier requirements.
                - inst_gen (instance_generator): An instance generator object.

                Returns:
                tuple: A tuple containing the generated route, total distance, total load, and updated list of pending suppliers.

                Example:
                ```python
                route, distance, load, pending_sup = Routing.RCL_constructive.generate_RCL_route(0.35, pending_sup,
                                                                                                requirements, inst_gen)
                ```
                """
                node:int = 0
                load:int = 0
                route:list = [node]
                distance:float = 0

                while load < inst_gen.Q:
                    target = Routing.RCL_constructive.generate_RCL_candidate(RCL_alpha,node,load,distance,pending_sup,requirements,inst_gen)
                    if target == False:
                        break
                    else:
                        load += requirements[target]
                        distance += inst_gen.c[node,target]
                        node = target
                        route.append(node)
                        pending_sup.remove(node)

                route.append(0)
                distance += inst_gen.c[node,0]

                return route,distance,load,pending_sup


        ''' Genetic Algorithm '''
        @staticmethod
        def GeneticAlgorithm(purchase:dict,inst_gen:instance_generator,t:int,return_top:int or bool=False,
                             time_limit:float=30,Population_size:int=1000,Elite_prop:float=0.25,
                             mutation_rate:float=0.5,verbose:bool=False,rd_seed=None):            
            start = process_time()
            if seed!=None:
                seed(rd_seed)
            pending_sup,requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            # Parameters
            Population_iter:list = [i for i in range(Population_size)]
            training_time:float = 3
            Elite_size:int = int(Population_size*Elite_prop)
            

            if verbose: print('Generating population')
            Population,FOs,Distances,Loads,incumbent,best_individual,alpha_performance = \
                            Routing.GA.generate_population(inst_gen,start,requirements,Population_iter,
                                                           training_time,t,verbose=False)
            
            # Print progress
            if verbose: 
                print('\n')
                print(f'----- Genetic Algorithm -----')
                print('\nt \tgen \tFO \t#V \tRoutes')
                print(f'{round(best_individual[3],2)} \t-1 \t{round(incumbent,2)} \t{len(best_individual[0])} \t{best_individual[0]}')

            # Genetic process
            generation = 0
            while process_time()-start <= time_limit:
                # print(f'generation {generation}')
                ### Elitism
                Elite = Routing.GA.elite_class(FOs, Population_iter, Elite_size)

                ### Selection: From a population, which parents are able to reproduce
                # Intermediate population: Sample of the initial population 
                inter_population = Routing.GA.intermediate_population(FOs, Population_size, Population_iter, Elite_size)            
                inter_population = Elite + list(inter_population)

                ### Tournament: Select two individuals and leave the best to reproduce
                Parents = Routing.GA.tournament(inter_population, FOs, Population_iter)

                ### Evolution
                New_Population:list = list();   New_FOs:list = list(); 
                New_Distances:list = list();   New_Loads:list = list()

                for i in Population_iter:
                    individual_i = Parents[i][randint(0,2)]
                    mutated = False

                    ###################
                    # Operators
                    ###################
                    new_individual,new_distances,evolved = Routing.GA.mutation(Population[individual_i],
                                                                        Distances[individual_i],inst_gen,mutation_rate)
                    new_FO = sum(new_distances) 
                    new_loads = Loads[individual_i]
                    mutated = True

                    
                    # No operation is performed
                    if not mutated: 
                        new_individual = Population[individual_i];new_FO = FOs[individual_i] 
                        new_distances = Distances[individual_i];new_loads = Loads[individual_i]

                    # Store new individual
                    New_Population.append(new_individual);New_FOs.append(new_FO); 
                    New_Distances.append(new_distances);New_Loads.append(new_loads)

                    # Updating incumbent
                    if new_FO < incumbent:
                        incumbent = deepcopy(new_FO)
                        best_individual = [deepcopy(new_individual),deepcopy(new_FO),(deepcopy(new_distances),deepcopy(new_loads)),process_time()-start]
                        # print(f'{round(process_time() - start)} \t{generation} \t{incumbent} \t{len(new_individual)} \t{best_individual[0]}')

                # Update population
                Population = New_Population
                FOs = New_FOs
                Distances = New_Distances
                Loads = New_Loads
                generation += 1

            
            if verbose:
                print('\n')

            if not return_top:
                # print('\n')
                # print(best_individual[0])
                return *best_individual,None
            else:
                combined = list(zip(Distances, Population))                 # Combine 'Distances' and 'Population' into tuples
                sorted_combined = sorted(combined, key=lambda x: x[0])      # Sort the combined list based on the 'Distances' values
                # Extract the five elements with the lowest unique distances
                result = []
                unique_distances = set()
                for entry in sorted_combined:
                    distance, element = entry
                    if distance not in unique_distances:
                        result.append(element)
                        unique_distances.add(distance)
                        if len(result) == 5:
                            break

                return *best_individual,result

        class GA():
            ''' Generate initial population '''
            @staticmethod
            def generate_population(inst_gen:instance_generator,start:float,requirements:dict,Population_iter:range,
                                    training_time:float,t,verbose:bool):
                # Initalizing data storage
                Population = list()
                FOs = list()
                Distances = list()
                Loads = list()

                incumbent:float = 1e9
                best_individual:list = list()
                
                # Adaptative-Reactive Constructive
                # RCL_alpha_list:list = [0,0.001,0.005]
                RCL_alpha_list:list = [0.05, 0.15, 0.25, 0.4]
                alpha_performance:dict = {alpha:0 for alpha in RCL_alpha_list}

                # Calibrating alphas
                training_ind:int = 0
                while process_time()-start<=training_time:
                    alpha_performance = Routing.GA.calibrate_alpha(RCL_alpha_list,alpha_performance,requirements,inst_gen)
                    training_ind += 1
                
                # Including an individual generated by the NN heuristic
                requirements2 = deepcopy(requirements)
                individual,FO,(distances,loads),_ = Routing.NearestNeighbor(requirements2,inst_gen,t)
                Population.append(individual);FOs.append(FO);Distances.append(distances);Loads.append(loads)
                if FO < incumbent:
                    incumbent = FO
                    best_individual: list = [individual,FO,(distances,loads),process_time()-start]

                # Generating initial population
                for ind in Population_iter[:-1]:
                    requirements2 = deepcopy(requirements)
                    
                    # Choosing alpha
                    RCL_alpha = choice(RCL_alpha_list, p = [alpha_performance[alpha]/sum(alpha_performance.values()) for alpha in RCL_alpha_list])    
                
                    # Generating individual
                    individual,FO,(distances,loads), _ = Routing.RCL_Solution(requirements2,inst_gen,0,RCL_alpha)
                    
                    if verbose: 
                        print(f'Generated individual {ind}')
                    # Updating incumbent
                    if FO < incumbent:
                        incumbent = FO
                        best_individual: list = [individual,FO,(distances,loads),process_time()-start]

                    # Saving individual
                    Population.append(individual)
                    FOs.append(FO)
                    Distances.append(distances)
                    Loads.append(loads)

                return Population,FOs,Distances,Loads,incumbent,best_individual,alpha_performance


            ''' Calibrate alphas for RCL '''
            @staticmethod
            def calibrate_alpha(RCL_alpha_list:list,alpha_performance:dict,requirements:dict,inst_gen:instance_generator) -> dict:
                requirements2 = deepcopy(requirements)
                tr_distance:float = 0
                RCL_alpha:float = choice(RCL_alpha_list)
                routes, FO, (distances, loads), _ = Routing.RCL_Solution(requirements2,inst_gen,0,RCL_alpha)
                tr_distance += FO
                alpha_performance[RCL_alpha] += 1/tr_distance

                return alpha_performance


            ''' Elite class '''
            @staticmethod
            def elite_class(FOs:list, Population_iter:range, Elite_size:int) -> list:
                return [x for _, x in sorted(zip(FOs,[i for i in Population_iter]))][:Elite_size] 
            

            ''' Intermediate population '''
            @staticmethod
            def intermediate_population(FOs:list, Population_size:int, Population_iter:range, Elite_size:int):
                # Fitness function
                # Fitness function
                tots = sum(FOs)
                fit_f:list = list()
                probs:list = list()
                for i in Population_iter:
                    fit_f.append(tots/FOs[i])
                for i in Population_iter:
                    probs.append(fit_f[i]/sum(fit_f))
                return choice([i for i in Population_iter],size=int(Population_size - Elite_size),replace=True,p=probs)


            ''' Tournament '''
            @staticmethod
            def tournament(inter_population:list, FOs:list, Population_iter:range) -> list:
                Parents:list = list()
                for i in Population_iter:
                    parents:list = list()
                    for j in range(2):
                        candidate1 = choice(inter_population);  val1 = FOs[candidate1]
                        candidate2 = choice(inter_population);  val2 = FOs[candidate2]

                        if val1 < val2:     parents.append(candidate1)
                        else:               parents.append(candidate2)
                    
                    Parents.append(parents)

                return Parents


            @staticmethod
            def mutation(individual:list,distances:float,inst_gen:instance_generator,mutation_rate:float)->list:
                pos = randint(0,len(individual))
                if len(individual[pos])<=3: 
                    return individual,distances,False
                
                if random() <= mutation_rate:
                    individual[pos],distances[pos],mutated = Routing.GA.swap_mutation(individual[pos],distances[pos],inst_gen.c,d_max=inst_gen.d_max,inst_gen=inst_gen)
                else:
                    individual[pos],distances[pos],mutated = Routing.GA.two_opt_mutation(individual[pos],distances[pos],inst_gen.c,d_max=inst_gen.d_max)
                
                return individual,distances,mutated


            @staticmethod
            def swap_mutation(route:list,current_obj:float,c:dict,**kwargs)->tuple:
                """
                Performs a SWAP mutation on a route. Two positions are randomly selected
                and their values are swapped. Returns the mutated route and its new distance.

                Parameters:
                route (list): The route to mutate.
                distance_matrix (dict): Distance matrix with distances between nodes.

                Returns:
                tuple: The mutated route and its new distance.
                """
                size = len(route)
                idx1, idx2 = sorted(np.random.choice(range(1,size-1),2,replace=False))
                
                if idx2 == idx1+1: return route,current_obj,False
                # Swap the nodes
                old_route = deepcopy(route)
                route[idx1],route[idx2] = route[idx2],route[idx1]

                # Recompute the affected distances
                delta_distance = 0
                
                delta_distance += (c[route[idx1 - 1], route[idx1]]
                  - c[route[idx1 - 1], route[idx2]])

                delta_distance += (c[route[idx1], route[idx1 + 1]]
                                - c[route[idx2], route[idx1 + 1]])

                # Ensure that idx2 + 1 is within the valid range
                if idx2 + 1 < size:
                    delta_distance += (c[route[idx2], route[idx2 + 1]]
                                    - c[route[idx1], route[idx2 + 1]])

                # Ensure that idx2 - 1 is within the valid range
                if idx2 - 1 >= 0:
                    delta_distance += (c[route[idx2 - 1], route[idx2]]
                                    - c[route[idx2 - 1], route[idx1]])

                new_distance = current_obj+delta_distance

                # Evaluate route's feasibility
                if new_distance > kwargs['d_max']:
                    return old_route,current_obj,False
                
                return route,new_distance,True
        

            @staticmethod
            def two_opt_mutation(route,current_obj,c,**kwargs):
                """
                Applies a 2-opt mutation on a route's segment. The segment is reversed, which might lead to a shorter path.
                Returns the mutated route and its new distance. Ensures the route starts and ends with the depot (0).

                Parameters:
                route (list): The route to mutate.
                c (dict): Distance matrix with distances between nodes.
                current_obj (float): The current total distance of the route.

                Returns:
                tuple: The mutated route and its new distance.
                """
                size = len(route)
                
                # Ensure that the depot is not included in the reversed segment
                idx1,idx2 = sorted(np.random.choice(range(1,size - 1), 2,replace=False))

                # Reverse the segment
                new_route = route[:idx1] + route[idx1:idx2][::-1] + route[idx2:]

                # Recompute the affected distances
                delta_distance = 0
                if idx1 > 0:
                    delta_distance += (c[new_route[idx1 - 1], new_route[idx1]] 
                                    - c[route[idx1 - 1], route[idx1]])
                if idx2 < size - 1:
                    delta_distance += (c[new_route[idx2 - 1], new_route[idx2]] 
                                    - c[route[idx2 - 1], route[idx2]])

                new_distance = current_obj + delta_distance
                
                # Evaluate route's feasibility
                if 'd_max' in kwargs and new_distance > kwargs['d_max']:
                    return route,current_obj,False

                return new_route,new_distance,True


        ''' Hybrid Genetic Search (CVRP) '''
        class HyGeSe(): 
            # Generate routes
            @staticmethod
            def HyGeSe_routing(purchase:dict,inst_gen:instance_generator,t:int,time_limit:int=30):    
                start = process_time()
                # Solver initialization
                ap = hgs.AlgorithmParameters(timeLimit=time_limit)  # seconds
                hgs_solver = hgs.Solver(parameters=ap,verbose=False)

                pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)
                loading_time = process_time()-start

                data = Routing.HyGeSe.generate_HyGeSe_data(inst_gen, requirements)

                # Save the original stdout
                result = hgs_solver.solve_cvrp(data)
                time = result.time

                routes = Routing.HyGeSe.translate_routes(inst_gen,requirements,result.routes)

                return routes,result.cost,time


            # Generate data dict for HyGeSe algorithm
            @staticmethod
            def generate_HyGeSe_data(inst_gen:instance_generator,requirements:dict)->dict:
                data = dict()

                # data['distance_matrix'] = [[inst_gen.c[i,j] if i!=j else 0 for j in inst_gen.N] for i in inst_gen.N]
                data['distance_matrix'] = [[inst_gen.c[i,j] if i!=j else 0 for j in inst_gen.V if (j in requirements.keys() or j==0)] for i in inst_gen.V if (i in requirements.keys() or i==0)]
                data['demands'] = np.array([0] + list(requirements.values()))
                data['vehicle_capacity'] = inst_gen.Q
                data['num_vehicles'] = inst_gen.F
                data['depot'] = 0
            
                return data

            # Translate routes to environment suppliers
            @staticmethod
            def translate_routes(inst_gen:instance_generator,requirements:dict,routes:list[list]):
                dic = {i+1:val for i,val in enumerate(list(requirements.keys()))}
                new_routes = list()
                for route in routes:
                    new_route = [0]
                    for sup in route:
                        new_route.append(dic[sup])
                    new_route.append(0)
                    new_routes.append(new_route)
                return new_routes
                        

        ''' Mixed Integer Program '''
        @staticmethod
        def MixedIntegerProgram(purchase:dict,inst_gen:instance_generator,t:int):
            start = process_time()
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)
    
            N, V, A, distances, requirements = Routing.network_aux_methods.generate_complete_graph(inst_gen, pending_sup, requirements)
            A.append((0,inst_gen.M+1))
            distances[0,inst_gen.M+1] = 0

            model = Routing.MIP.generate_complete_MIP(inst_gen, N, V, A, distances, requirements)

            model.update()
            model.setParam('OutputFlag',0)
            model.setParam('MIPGap',0.0001)
            model.setParam('TimeLimit',7200)
            model.optimize()

            routes, distances, loads = Routing.MIP.get_MIP_decisions(inst_gen, model, V, A, distances, requirements)
            cost = model.getObjective().getValue()

            return routes, cost, (distances, loads), process_time() - start    
        
        class MIP():
            # Generate complete MIP model
            @staticmethod
            def generate_complete_MIP(inst_gen:instance_generator,N:list,V:range,A:list,distances:dict,requirements:dict)->gu.Model:
                model = gu.Model('d-CVRP')

                # 1 if arch (i,j) is traveled by vehicle f, 0 otherwise
                x = {(i,j,f):model.addVar(vtype = gu.GRB.BINARY, name = f'x_{i}{j}{f}') for (i,j) in A for f in inst_gen.Vehicles}
                
                # Cumulative distance until node i by vehicle f
                w = {(i,f):model.addVar(vtype = gu.GRB.CONTINUOUS, name = f'w_{i}{f}') for i in V for f in inst_gen.Vehicles}

                # 2. Every node is visited
                for i in N:
                    model.addConstr(gu.quicksum(x[i,j,f] for f in inst_gen.Vehicles for j in V if (i,j) in A) == 1)

                for f in inst_gen.Vehicles:
                    # 3. All vehicles start at depot
                    model.addConstr(gu.quicksum(x[0,j,f] for j in V if (0,j) in A) == 1)

                    # 4. All vehicles arrive at depot
                    model.addConstr(gu.quicksum(x[i,inst_gen.M+1,f] for i in V if (i,inst_gen.M+1) in A) == 1)

                    # 5. Flow preservation
                    for i in N:
                        model.addConstr(gu.quicksum(x[i,j,f] for j in V if (i,j) in A) - gu.quicksum(x[j,i,f] for j in V if (j,i) in A) == 0)

                    # 6. Max distance per vehicle
                    model.addConstr(gu.quicksum(distances[i,j]*x[i,j,f] for (i,j) in A) <= inst_gen.d_max)

                    # 7. Max capacity per vehicle
                    model.addConstr(gu.quicksum(requirements[i] * gu.quicksum(x[i,j,f] for j in V if (i,j) in A) for i in V) <= inst_gen.Q)

                    # 8. Distance tracking/No loops
                    for (i,j) in A:
                        model.addConstr(w[i,f] + distances[i,j] - w[j,f] <= (1 - x[i,j,f])*1e4)

                total_distance = gu.quicksum(distances[i,j] * x[i,j,f] for (i,j) in A for f in inst_gen.Vehicles)
                model.setObjective(total_distance)
                model.modelSense = gu.GRB.MINIMIZE

                return model
            

            # Retrieve and consolidate decisions from MIP
            @staticmethod
            def get_MIP_decisions(inst_gen:instance_generator,model:gu.Model,V:list,A:list,distances:dict,requirements:dict):
                routes = list()
                dist = list()
                loads = list()

                for f in inst_gen.Vehicles:
                    node = 0
                    route = [node]
                    distance = 0
                    load = 0
                    while True:
                        for j in V:
                            if (node,j) in A and model.getVarByName(f'x_{node}{j}{f}').x > 0.5:
                                route.append(j)
                                distance += distances[node,j]
                                load += requirements[j]
                                node = j
                                break

                        if node == inst_gen.M+1:
                            del route[-1]
                            route.append(0)
                            if route != [0,0]:
                                routes.append(route)
                                dist.append(distance)
                                loads.append(load)
                            break

                return routes,dist,loads
        

        ''' Column generation algorithm '''
        @staticmethod
        def ColumnGeneration(purchase:dict,inst_gen:instance_generator,t:int,heuristic_initialization=False,
                             time_limit=False,verbose:bool=False,return_num_cols:bool=False,RCL_alpha:float=0.35)->tuple:
            start = process_time()
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            N, V, A, distances, requirements = Routing.network_aux_methods.generate_complete_graph(inst_gen,pending_sup,requirements)
            sup_map = {i:(idx+1) for idx,i in enumerate(N)}

            master = Routing.Column_Generation.MasterProblem()
            modelMP,theta,RouteLimitCtr,NodeCtr,objectives,cols,loadss = master.buidModel(inst_gen,N,distances,t,heuristic_initialization,requirements=requirements,RCL_alpha=RCL_alpha)

            card_omega = len(theta)    # number of routes
            init_cols = max(deepcopy(card_omega)-len(N),0)
            same_objective_count = 0
            last_objective_value = None

            iter = 0
            opt_flag = False

            
            routes = list()
            loads = list()
            for i in N:
                routes.append([0,i,0])
                loads.append(requirements[i])
            routes+=cols
            loads+=loadss

            if verbose:
                print("--------- Column Generation Algorithm ---------\n",flush = True)
                print('\t \t|    Relaxed/Restr MP \t|    Auxiliary Problem')
                print('Iter \ttime \t| MP_FO \t#Veh \t| r.c \t \tRoute')
                print('------------------------------------------------------------------')
            
            while True:
                # print('Solving Master Problem (MP)...', flush = True)
                iter += 1
                modelMP.optimize()
                current_objective_value = modelMP.getObjective().getValue()
                if verbose:
                    print(f'{iter} \t{round(process_time()-start,2)} \t| {round(current_objective_value,2)} \t{sum(list(modelMP.getAttr("X", modelMP.getVars())))}',end='\r')
                
                lambdas = list()
                lambdas.append(modelMP.getAttr("Pi",RouteLimitCtr)[0])
                lambdas += modelMP.getAttr("Pi",NodeCtr)
                
                # Check stopping criterion
                if (last_objective_value is not None and current_objective_value == last_objective_value):
                    same_objective_count += 1
                else:
                    same_objective_count = 0

                if same_objective_count >= 100:
                    if verbose:
                        print("\nStoping criterion: Number of iterations without change", flush=True)
                    break

                last_objective_value = current_objective_value

                a_star = dict()
                a_star.update({i:0 for i in N})
                shortest_path,a_star,sol = Routing.Column_Generation.SubProblem.solveAPModel(
                                                        lambdas,a_star,inst_gen,N,V,A,distances,requirements,sup_map)
                routes.append(sol[0]); objectives.append(sol[1]); loads.append(sol[2])
                minReducedCost = shortest_path[0]
                c_k = shortest_path[1]

                # Check termination condition
                if  minReducedCost >= -0.00005:
                    if verbose:
                        opt_flag = True
                        print("\nStoping criterion: % gap", flush = True)
                    break
                elif time_limit and process_time() - start > time_limit:
                    if verbose:
                        print("\nStoping criterion: time", flush = True)
                    break
                else:
                    if verbose:
                        print(f'{iter} \t{round(process_time()-start,2)} \t| {round(current_objective_value,2)} \t{round(sum(list(modelMP.getAttr("X", modelMP.getVars()))),2)} \t| {round(minReducedCost,2)} \t{sol[0]}')
                    # print('Minimal reduced cost (via CSP):', minReducedCost, '<0.', flush = True)
                    a_star = list(a_star.values())
                    a_star.append(1) #We add the 1 of the number of routes restrictions

                    newCol = gu.Column(a_star,modelMP.getConstrs())
                    theta.append(modelMP.addVar(vtype=gu.GRB.CONTINUOUS,obj=c_k,lb=0,
                                                column=newCol,name=f"theta_{card_omega}"))
                    card_omega+=1
                    # Update master model
                    modelMP.update()
            
            for v in modelMP.getVars():
                v.setAttr("Vtype", gu.GRB.INTEGER)

            modelMP.optimize()

            if verbose:
                print('Integer Master Problem:')
                print(f'Objective: {round(modelMP.objVal,2)}')
                print(f'Vehicles:  {sum(list(modelMP.getAttr("X", modelMP.getVars())))}')
                print(f'Time:       {round(process_time()-start,2)}s')
                print('Normal termination. -o-')
            
            routes,distances,loads = Routing.Column_Generation.decode_CG_routes(inst_gen,modelMP,routes,objectives,loads)
            
            if not return_num_cols:
                return routes,sum(distances),(distances,loads,opt_flag),process_time()-start
            else:
                return routes,sum(distances),(distances,loads,opt_flag),process_time()-start,(init_cols,len(theta)-len(N)-init_cols)
        
        class Column_Generation():
            # Master problem
            class MasterProblem:

                def __init__(self):
                    pass

                def buidModel(self,inst_gen:instance_generator,N:list,distances:dict,t:int,heuristic_initialization,name:str='MasterProblem',**kwargs):
                    modelMP = gu.Model(name)

                    modelMP.Params.OutputFlag = 0
                    modelMP.setParam('Presolve', 0)
                    modelMP.setParam('Cuts', 0)

                    modelMP,theta,objectives = self.generateVariables(inst_gen,modelMP,N,distances,t)
                    modelMP,RouteLimitCtr,NodeCtr = self.generateConstraints(inst_gen, modelMP, N, theta)

                    if heuristic_initialization != False:
                        modelMP,theta,objectives,cols,loadss = self.heuristic_initialization(modelMP,N,kwargs['requirements'],inst_gen,t,
                                                                                             theta,objectives,kwargs['RCL_alpha'],init_time=heuristic_initialization)
                    modelMP = self.generateObjective(modelMP)
                    modelMP.update()

                    if not heuristic_initialization:
                        return modelMP,theta,RouteLimitCtr,NodeCtr,objectives,list(),list()
                    else:
                        return modelMP,theta,RouteLimitCtr,NodeCtr,objectives,cols,loadss
            

                def generateVariables(self,inst_gen:instance_generator,modelMP:gu.Model,N:list,distances,t:int):
                    theta = list()
                    objectives = list()
                    
                    for idx,i in enumerate(N):
                        route_cost = distances[0,i] + distances[i,inst_gen.M+1]
                        theta.append(modelMP.addVar(vtype=gu.GRB.CONTINUOUS,obj=route_cost,lb=0,name=f"theta_{idx}"))
                        objectives.append(route_cost)

                    

                    return modelMP,theta,objectives


                def generateConstraints(self,inst_gen:instance_generator,modelMP:gu.Model,N:list,theta:list):
                    NodeCtr = list()                #Node covering constraints
                    for idx,i in enumerate(N):
                        NodeCtr.append(modelMP.addConstr(theta[idx]>=1, f"Set_Covering_{i}")) #Set covering constraints
                    
                    RouteLimitCtr = list()          #Limits the number of routes
                    RouteLimitCtr.append(modelMP.addConstr(gu.quicksum(theta[i] for i in range(len(theta))) <= inst_gen.F, 'Route_Limit_Ctr')) #Routes limit Constraints

                    modelMP.update()

                    return modelMP, RouteLimitCtr, NodeCtr


                def generateObjective(self,modelMP:gu.Model):
                    modelMP.modelSense = gu.GRB.MINIMIZE
                    return modelMP 


                def heuristic_initialization(self,modelMP:gu.Model,N,requirements:dict,inst_gen:instance_generator,t,theta:list,objectives:list,RCL_alpha:float,init_time:float=5):
                    cols = list()
                    loadss = list()
                    HI_start = process_time()

                    card_omega = len(theta)

                    new_req = {key:value for key,value in requirements.items() if key not in [0,inst_gen.M+1]}
                    while process_time()-HI_start < init_time:
                        routes,FO,info,time = Routing.RCL_Solution(new_req,inst_gen,t,RCL_alpha=RCL_alpha)

                        for i,route in enumerate(routes):
                            if route not in cols and len(route) > 5:
                                card_omega += 1
                                route_cost = info[0][i]

                                a_star = [1 if j in route else 0 for j in N]
                                a_star.append(1) # Add the 1 for the 'Number of vehicles' constraint

                                newCol = gu.Column(a_star,modelMP.getConstrs())
                                theta.append(modelMP.addVar(vtype=gu.GRB.CONTINUOUS,obj=route_cost,lb=0,
                                                            column=newCol,name=f"theta_{card_omega}"))
                                
                                objectives.append(route_cost)

                                cols.append(route)
                                loadss.append(info[1][i])
                    
                    modelMP.update()

                    return modelMP,theta,objectives,cols,loadss

            # Auxiliary problem
            class SubProblem:
                
                @staticmethod
                def solveAPModel(lambdas,a_star,inst_gen,N,V,A,distances,requirements,sup_map):
                    
                    modelAP = gu.Model('SubProblem')
                    modelAP.Params.OutputFlag = 0
                    modelAP.Params.BestObjStop = -0.01

                    x = dict()                          #Flow variables     
                    for (i,j) in A:
                        x[i,j] = modelAP.addVar(vtype = gu.GRB.BINARY, lb = 0, name = f"x_{i},{j}")
                    w = dict()        
                    for i in V:
                        w[i] = modelAP.addVar(vtype = gu.GRB.CONTINUOUS, name = f"w_{i}")
            
                    # 3. All vehicles start at depot
                    modelAP.addConstr(gu.quicksum(x[0,j] for j in N if (0,j) in A) == 1, "Depart from depot")

                    # 4. All vehicles arrive at depot
                    modelAP.addConstr(gu.quicksum(x[i,inst_gen.M+1] for i in V if (i,inst_gen.M+1) in A) == 1, "Reach the depot")

                    # 5. Flow preservation
                    for i in N:
                        modelAP.addConstr(gu.quicksum(x[i,j] for j in V if (i,j) in A) - gu.quicksum(x[j,i] for j in V if (j,i) in A) == 0, f'Flow conservation_{i}')

                    # 6. Max distance per vehicle
                    modelAP.addConstr(gu.quicksum(distances[i,j]*x[i,j] for (i,j) in A) <= inst_gen.d_max, 'Max distance')

                    # 7. Max capacity per vehicle
                    modelAP.addConstr(gu.quicksum(requirements[i]*gu.quicksum(x[i,j] for j in V if (i,j) in A) for i in V) <= inst_gen.Q, "Capacity")

                    # 8. Distance tracking/No loops
                    for (i,j) in A:
                        modelAP.addConstr(w[i]+distances[i,j]-w[j] <= (1-x[i,j])*1e4, f'Distance tracking_{i}{j}')

                    #Shortest path objective
                    c_trans = dict()
                    for (i,j) in A:
                        if i in N:
                            c_trans[i,j] = distances[i,j]-lambdas[sup_map[i]]
                        else:
                            c_trans[i,j] = distances[i,j]
                    
                    modelAP.setObjective(sum(c_trans[i,j]*x[i,j] for (i,j) in A), gu.GRB.MINIMIZE)
                    # modelAP.setParam('OutputFlag',1)
                    modelAP.update()
                    modelAP.optimize()

                    route_cost = sum(distances[i,j]*x[i,j].x for (i,j) in A)

                    for i in N:
                        for j in V:
                            if (i,j) in A and x[i,j].x>0.5:
                                a_star[i] = 1

                    node = 0
                    route = [node]
                    load = 0
                    route_complete = False
                    while not route_complete:
                        for j in V:
                            if (node,j) in A and x[node,j].x>0.5:
                                route.append(j)
                                load+=requirements[j]
                                node = j
                                if j == inst_gen.M+1:
                                    route_complete = True
                                    route = route[:-1]+[0]
                                    break

                    return [modelAP.objVal,route_cost],a_star,[route,route_cost,load]

            # Decode the routes generated by the CG algorihm
            @staticmethod
            def decode_CG_routes(inst_gen:instance_generator,modelMP:gu.Model,routes:list,objectives:list,loads:list):
                solution = list()
                objective = list()
                load = list()
                for i,theta in enumerate(modelMP.getVars()):
                    if theta.x != 0:
                        solution.append(routes[i])
                        objective.append(objectives[i])
                        load.append(loads[i])
                
                return solution,objective,loads
                        

        ''' Pricing algorithm '''
        @staticmethod
        def route_pricing(inst_gen:instance_generator,new_route:list,purchase:dict,t:int,solution=None):
            pending_sup,requirements = Routing.consolidate_purchase(purchase,inst_gen,t)
            N,V,A,distances,requirements = Routing.network_aux_methods.generate_complete_graph(inst_gen,pending_sup,requirements)
            sup_map = {i:(idx+1) for idx,i in enumerate(N)}
            
            ### MASTER PROBLEM
            master = Routing.Column_Generation.MasterProblem()
            modelMP,theta,RouteLimitCtr,NodeCtr,objectives,_,_ = master.buidModel(inst_gen,N,distances,t,False)
            card_omega = len(theta)
            
            if solution != None:
                for route in solution:
                    # route.sort()
                    a_star = [1 if i in route else 0 for i in N]
                    a_star.append(1) # Add the 1 for the 'Number of vehicles' constraint

                    _,route_cost,_ = Routing_management.evaluate_routes(inst_gen,[route],requirements)

                    newCol = gu.Column(a_star,modelMP.getConstrs())
                    theta.append(modelMP.addVar(vtype=gu.GRB.CONTINUOUS,obj=route_cost,lb=0,
                                                column=newCol,name=f"theta_{card_omega}"))
                    card_omega+=1
                    # Update master model
                    modelMP.update()

            modelMP.optimize()
                
            lambdas = list()
            lambdas.append(modelMP.getAttr("Pi",RouteLimitCtr)[0])
            lambdas += modelMP.getAttr("Pi",NodeCtr)

            # a_star = dict()
            # a_star.update({i:0 for i in N})

            ### ROUTE PRICING
            c_trans = dict()
            for (i,j) in A:
                if i in N:
                    c_trans[i,j] = distances[i,j]-lambdas[sup_map[i]]
                else:
                    c_trans[i,j] = distances[i,j]
            
            reduced_cost = 0  
            for i,node in enumerate(new_route[:-2]):
                reduced_cost += c_trans[node,new_route[i+1]]
            reduced_cost += c_trans[new_route[-2],inst_gen.M+1]

            return reduced_cost


        @staticmethod
        def evaluate_randomized_policy(router,purchase,inst_gen:instance_generator,env, n=30,averages=True,
                                       dynamic_p=False,**kwargs)->tuple:
            times = list()
            vehicles = list()
            objectives = list()
            extra_costs = list()
            missings = list()

            if router == Routing.RCL_Heuristic:
                for i in range(n):
                    seed = (i+1) * 2
                    RCL_routes,RCL_obj,RCL_info,RCL_time = router(purchase,inst_gen,env.t,RCL_alphas=kwargs['RCL_alphas'],
                                                                  adaptative=kwargs['adaptative'],rd_seed=seed,time_limit=kwargs['time_limit'])
                    # assert abs(RCL_obj-Routing_management.price_routes(inst_gen,RCL_routes,purchase))<0.0001,"Computed distance doesn't match route cost"
                    times.append(RCL_time)
                    vehicles.append(len(RCL_routes))
                    objectives.append(RCL_obj)

                    if dynamic_p:
                        extra_cost,missing = Routing_management.evaluate_dynamic_potential(inst_gen,env,RCL_routes,purchase,
                                                                                            discriminate_missing=False)
                        extra_costs.append(extra_cost)
                        missings.append(missing)

            if dynamic_p:
                if averages:
                    return (np.mean(objectives),round(np.mean(vehicles), 2),np.mean(times),
                            np.std(objectives),np.min(objectives),np.max(objectives),
                            np.std(vehicles),np.min(vehicles),np.max(vehicles),
                            np.std(times),np.min(times),np.max(times),
                            np.mean(extra_costs),np.min(extra_costs),np.max(extra_costs),
                            np.mean(missings),np.min(missings),np.max(missings))
                else:
                    return objectives,vehicles,times,extra_costs,missings
            else:
                if averages:
                    return np.mean(objectives),round(np.mean(vehicles),2),np.mean(times),(np.median(objectives),
                           np.std(objectives),np.min(objectives),np.max(objectives))
                else:
                    return objectives,vehicles,times


        @staticmethod
        def multiprocess_eval_rand_policy(router,purchase,inst_gen:instance_generator,env, n=30,averages=True,
                                       dynamic_p=False,initial_seed=0,**kwargs):
            freeze_support()

            seeds = [i for i in range(initial_seed,initial_seed+n)] 
            p = pool.Pool()

            def run_eval(seed):
                RCL_routes,RCL_obj,RCL_info,RCL_time = router(purchase,inst_gen,env.t,RCL_alphas=kwargs['RCL_alphas'],
                                                            adaptative=kwargs['adaptative'],rd_seed=seed,
                                                            time_limit=kwargs['time_limit'])
                return RCL_routes,RCL_obj,RCL_info,RCL_time                
            
            Results = p.map(run_eval,seeds)
            p.close()
            p.join()

            objectives = [i[1] for i in Results]
            vehicles = [len(i[0]) for i in Results]
            times = [i[3] for i in Results]
            
            return np.mean(objectives),round(np.mean(vehicles),2),np.mean(times),(np.median(objectives),
                        np.std(objectives),np.min(objectives),np.max(objectives))





        ''' Auxiliary methods for network management '''
        class network_aux_methods():
            # Generate vertices and arches for a complete graph
            @staticmethod
            def generate_complete_graph(inst_gen:instance_generator,pending_sup:list,requirements:dict) -> tuple:
                N = pending_sup
                if 0 in N:
                    N.remove(0)
                if inst_gen.M+1 in N:
                    N.remove(inst_gen.M+1)
                V:list = [0]+ N +[inst_gen.M+1]
                A:list = [(i,j) for i in V for j in V if i!=j and i!=inst_gen.M+1 and j!=0 and not (i == 0 and j == inst_gen.M+1)]

                coors = deepcopy(inst_gen.coor)
                coors.update({(inst_gen.M+1):inst_gen.coor[0]})
                distances = dict()
                distances.update({(i,j):((coors[j][0]-coors[i][0])**2+(coors[j][1]-coors[i][1])**2)**(1/2) for (i,j) in A})
                # for i in V:
                #     for j in V:
                #         if (i,j) in A:
                #             x1, y1 = coors[i]; x2, y2 = coors[j]
                #             distances[i,j] = ((x2-x1)**2+(y2-y1)**2)**(1/2)
                
                requirements[0] = 0
                requirements[inst_gen.M+1] = 0

                return N, V, A, distances, requirements


        ''' Auxiliary method to compute total product to recover from suppliers '''
        @staticmethod
        def consolidate_purchase(purchase:dict,inst_gen:instance_generator,t:int)->tuple:
            # purchse is given for suppliers and products
            if type(list(purchase.keys())[0]) == tuple:
                pending_suppliers = list()
                requirements = dict()
                for i in inst_gen.Suppliers:
                    req = sum(purchase[i,k] for k in inst_gen.K_it[i,t] if (i,k) in purchase.keys())
                    if req > 0:
                        pending_suppliers.append(i)
                        requirements[i] = req

                return pending_suppliers,requirements
            # purchase is given for products
            else:
                return list(purchase.keys()), purchase


        class RouteMemoryAgent():

            def __init__(self,sol_num:int=100)->None:
                self.sol_num=sol_num

                self.solutions = [] 
                self.objectives = []

                self.solution_num = 0


            def update_pool(self,solution:list,objective:float):
                sorted_sol = sorted(solution,key=len)
                exists = False
                for i,sol in enumerate(self.solutions):
                    if self.are_equal(sol,i,sorted_sol,objective):          # Solutions are the same
                        exists = True
                    
                if not exists and len(self.solutions) < self.sol_num:       
                    self.solutions.append(solution)
                    self.objectives.append(objective)
                    self.solution_num += 1
                        

            def are_equal(self,routes,i,solution,objective):
                if objective != self.objectives[i] or len(routes) != len(solution): # Objectives or number of routes are different
                    return False
                

class RoutingAgent(Routing):

    def __init__(self,policies=['NN','RCL','GA','HGS','CG','MIP']):
        self.policies = policies


    def tune_policy(self,policy,grid:dict,inst_gen,num_episodes):
        if policy == 'RCL':
            for RCL_alpha in grid['RCL alpha']:
                for adaptative in grid['adaptative']:
                    pass


    def policy_routing(self,policy,purchase,inst_gen,t,**kwargs)->tuple:
        price_routes = False
        if 'price_routes' in kwargs.keys(): price_routes = kwargs['price_routes']

        time_limit = 120
        if 'time_limit' in kwargs.keys(): time_limit = kwargs['time_limit']

        if policy == 'NN':
            return self.NearestNeighbor(purchase,inst_gen,t,price_routes=price_routes,**kwargs)
        elif policy == 'RCL':
            return self.RCL_Heuristic(purchase,inst_gen,t,time_limit=time_limit,price_routes=price_routes,**kwargs)
        elif policy == 'GA':
            return self.GenticAlgorithm(purchase,inst_gen,t,time_limit=time_limit,**kwargs)
        elif policy == 'HGS':
            return self.HyGeSe(purchase,inst_gen,t,time_limit=time_limit,**kwargs)
        elif policy == 'CG':
            return self.ColumnGeneration(purchase,inst_gen,t,RCL_alpha=0.4,heuristic_initialization=5,time_limit=120)


    def random_policy(self,purchase:dict,inst_gen:instance_generator,t:int,**kwargs)->tuple:
        policy = choice(self.policies)

        return *self.policy_routing(policy,purchase,inst_gen,t,**kwargs),policy


    def get_best_action(self,state,q_table):
        state_t = tuple(state)
        dict_act = q_table[state_t]
        min_value = np.argmin(list(dict_act.values()))
        min_action = list(dict_act.keys())[min_value]
        action = list(min_action)
        return(action)
    

    def direct_shipping_cost(self,requirements:dict,inst_gen:instance_generator,)->float:
        return sum(inst_gen.c[0,i]*2 for i in requirements.keys())


class FlowerAgent(Routing):

    def __init__(self,solution_num:int=100):
        self.routes_num=solution_num

        self.routes = list()
        self.bincod = list()
        self.generator = list()
        self.metrics = list()
        self.history = list()
        self.n_table = list()

    def update_flower_pool(self,inst_gen,routes,generator,cost,total_SL,reactive_SL):
        sorted_routes = sorted(routes,key=lambda route:route[1])

        # New solution
        if sorted_routes not in self.routes:
            # Enough space
            if len(self.routes)<self.routes_num:
                self.routes.append(sorted_routes)
                self.bincod.append(self._code_binary_set_(inst_gen,routes))
                self.generator.append([generator])
                self.metrics.append([cost,cost/sum(self.bincod[-1]),total_SL,reactive_SL])
                self.history.append([[total_SL],[reactive_SL]])
                self.n_table.append(1)
            
            # Flower pool full
            else:
                pass
        
        # Already existing 
        else:
            index = self.routes.index(sorted_routes)
            if generator not in self.generator[index]:
                self.generator[index].append(generator)
            self.metrics[index][2] = self.metrics[index][2] + (1/self.n_table[index]) * (total_SL-self.metrics[index][2])
            self.metrics[index][3] = self.metrics[index][3] + (1/self.n_table[index]) * (reactive_SL-self.metrics[index][3])
            self.history[index][0].append(total_SL)
            self.history[index][1].append(reactive_SL)
            self.n_table[index]+=1


    def fit_purchase_to_existing_flower(self):
        pass
    
    
    def _code_binary_set_(self,inst_gen,routes):
        binary_encoding = np.zeros(inst_gen.M,dtype=int)
        for route in routes:
            for node in route[1:-1]:
                binary_encoding[node - 1] = 1  # Set the corresponding index to 1

        return binary_encoding


    def _compute_likeness(solution1, solution2):
        suppliers1 = set()
        suppliers2 = set()

        # Extract suppliers from the solutions
        for route in solution1:
            suppliers1.update(set(route[1:-1]))

        for route in solution2:
            suppliers2.update(set(route[1:-1]))

        # Calculate the Jaccard similarity index
        intersection = len(suppliers1.intersection(suppliers2))
        union = len(suppliers1.union(suppliers2))

        likeness_index = intersection / union if union != 0 else 0.0

        return likeness_index
    