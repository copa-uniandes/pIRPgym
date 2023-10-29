import pandas as np
import numpy as np
from numpy.random import seed,choice,randint
from time import time,process_time
from copy import deepcopy

import gurobi as gu

from ..InstanceGenerator import instance_generator


class Routing():
        options = ['NN, RCL, HGA, CG']
        
        ''' Nearest Neighbor (NN) heuristic '''
        # Generate routes
        def NearestNeighbor(purchase:dict[float],inst_gen:instance_generator,t:int) -> tuple:
            start = process_time()
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            routes:list = list()
            loads:list = list()
            t_distance:int = 0
            distances:list = list()

            while len(pending_sup) > 0:
                node: int = 0
                load: int = 0
                route: list = [node]
                distance: float = 0
                while load < inst_gen.Q:
                    target = Routing.Nearest_Neighbor.find_nearest_feasible_node(node, load, distance, pending_sup, requirements, inst_gen)
                    if target == False:
                        break
                    else:
                        load += requirements[target]
                        distance += inst_gen.c[node, target]
                        node = target
                        route.append(node)
                        pending_sup.remove(node)

                routes.append(route + [0])
                distance += inst_gen.c[node,0]
                loads.append(load)
                t_distance += distance
                distances.append(distance)
            
            return routes, distances, loads, process_time() - start
        
        class Nearest_Neighbor():      
            # Find nearest feasible (by capacity) node
            def find_nearest_feasible_node(node, load, distance, pending_sup, requirements, inst_gen):
                target, dist = False, 1e6
                for candidate in pending_sup:
                    if inst_gen.c[node,candidate] < dist and load + requirements[candidate] <= inst_gen.Q \
                        and distance + inst_gen.c[node,candidate] + inst_gen.c[candidate,0] <= inst_gen.d_max:
                        target = candidate
                        dist = inst_gen.c[node,target]
                
                return target
            
        
        ''' RCL based constructive '''
        def RCL_Heuristic(purchase:dict[float],inst_gen:instance_generator,t,RCL_alpha:float=0.35) -> dict:
            start = process_time()
            pending_sup, requirements = Routing.consolidate_purchase(purchase, inst_gen,t)

            routes:list = list()
            FO:int = 0
            distances:list = list()
            loads:list = list()
            dep_d_details:list = list() # TODO: Record departure times

            while len(pending_sup) > 0:
                route, distance, load, pending_sup = Routing.RCL_constructive.generate_RCL_route(RCL_alpha, pending_sup, requirements, inst_gen)

                routes.append(route)
                FO += distance
                distances.append(distance)
                loads.append(load)
                
            return routes, FO, distances, loads, process_time() - start
        
        class RCL_constructive():
            # Generate candidate from RCL
            def generate_RCL_candidate(RCL_alpha, node, load, distance, pending_sup, requirements, inst_gen):
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
            def generate_RCL_route(RCL_alpha, pending_sup, requirements, inst_gen):
                node:int = 0
                load:int = 0
                route:list = [node]
                distance:float = 0

                while load < inst_gen.Q:
                    target = Routing.RCL_constructive.generate_RCL_candidate(RCL_alpha, node, load, distance, pending_sup, requirements, inst_gen)
                    if target == False:
                        break
                    else:
                        load += requirements[target]
                        distance += inst_gen.c[node, target]
                        node = target
                        route.append(node)
                        pending_sup.remove(node)

                route.append(0)
                distance += inst_gen.c[node,0]

                return route, distance, load, pending_sup


        ''' Genetic Algorithm '''
        def HybridGenticAlgorithm(purchase:dict,inst_gen:instance_generator,t:int,top:int or bool=False,rd_seed:int=0,time_limit:float=30):
            start = process_time()
            seed(rd_seed)
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            # Parameters
            verbose = False
            Population_size:int = 5_000
            Population_iter:range = range(Population_size)
            training_time:float = 0.15*time_limit
            Elite_size:int = int(Population_size*0.25)

            crossover_rate:float = 0.5
            mutation_rate:float = 0.5

            Population, FOs, Distances, Loads, incumbent, best_individual, alpha_performance =\
                            Routing.GA.generate_population(inst_gen, start, requirements, verbose, 
                                                                            Population_iter, training_time)
            
            # Print progress
            if verbose: 
                print('\n')
                print(f'----- Genetic Algorithm -----')
                print('\nt \tFO \t#V \tgen')
                print(f'{round(best_individual[3],2)} \t{round(incumbent,2)} \t{len(best_individual[0])} \t-1')

            # Genetic process
            generation = 0
            while time()-start < time_limit:
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
                    # TODO: Operators
                    ###################

                    # No operator is performed
                    if not mutated: 
                        new_individual = Population[individual_i]; new_FO = FOs[individual_i] 
                        new_distances = Distances[individual_i]; new_loads = Loads[individual_i]

                    # Store new individual
                    New_Population.append(new_individual); New_FOs.append(new_FO); 
                    New_Distances.append(new_distances); New_Loads.append(new_loads)

                    # Updating incumbent
                    if sum(new_distances) < incumbent:
                        incumbent = sum(new_distances)
                        best_individual:list = [new_individual, new_distances, new_loads, process_time() - start]
                        print(f'{round(process_time() - start)} \t{incumbent} \t{len(new_individual)} \t{generation}')

                # Update population
                Population = New_Population
                FOs = New_FOs
                Distances = New_Distances
                Loads = New_Loads
                generation += 1
            
            if verbose:
                print('\n')

            if not top:
                return best_individual, None
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

                return best_individual, result

        class GA():
            ''' Generate initial population '''
            def generate_population(inst_gen:instance_generator, start:float, requirements:dict, verbose:bool, Population_iter:range,
                                    training_time: float):
                # Initalizing data storage
                Population:list[list] = list()
                FOs:list[float] = list()
                Distances:list[float] = list()
                Loads:list[float] = list()

                incumbent:float = 1e9
                best_individual:list = list()
                
                # Adaptative-Reactive Constructive
                RCL_alpha_list:list[float] = [0.15, 0.25, 0.35, 0.5, 0.75]
                alpha_performance:dict = {alpha:0 for alpha in RCL_alpha_list}

                # Calibrating alphas
                training_ind:int = 0
                while process_time() - start <= training_time:
                    alpha_performance = Routing.GA.calibrate_alpha(RCL_alpha_list, alpha_performance, requirements, inst_gen)
                    training_ind += 1
                
                # Generating initial population
                for ind in Population_iter:
                    requirements2 = deepcopy(requirements)
                    
                    # Choosing alpha
                    RCL_alpha = choice(RCL_alpha_list, p = [alpha_performance[alpha]/sum(alpha_performance.values()) for alpha in RCL_alpha_list])    
                
                    # Generating individual
                    individual, FO, distances, loads, _ = Routing.RCL_constructive.RCL_routing(requirements2, inst_gen, RCL_alpha)

                    # Updating incumbent
                    if FO < incumbent:
                        incumbent = FO
                        best_individual: list = [individual, distances, loads, process_time() - start]

                    # Saving individual
                    Population.append(individual)
                    FOs.append(FO)
                    Distances.append(distances)
                    Loads.append(loads)

                return Population, FOs, Distances, Loads, incumbent, best_individual, alpha_performance


            ''' Calibrate alphas for RCL '''
            def calibrate_alpha(RCL_alpha_list:list, alpha_performance:dict, requirements:dict, inst_gen:instance_generator) -> dict:
                requirements2 = deepcopy(requirements)
                tr_distance:float = 0
                RCL_alpha:float = choice(RCL_alpha_list)
                routes, FO, distances, loads, _ = Routing.RCL_constructive.RCL_routing(requirements2, inst_gen, RCL_alpha)
                tr_distance += FO
                alpha_performance[RCL_alpha] += 1/tr_distance

                return alpha_performance


            ''' Elite class '''
            def elite_class(FOs:list, Population_iter:range, Elite_size:int) -> list:
                return [x for _, x in sorted(zip(FOs,[i for i in Population_iter]))][:Elite_size] 
            

            ''' Intermediate population '''
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
                return choice([i for i in Population_iter], size = int(Population_size - Elite_size), replace = True, p = probs)


            ''' Tournament '''
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


        ''' Hybrid Genetic Search (CVRP) '''
        class HyGeSe(): 
            # Generate routes  
            def HyGeSe_routing(purchase:dict[float],inst_gen:instance_generator,t:int,time_limit:int=30):    
                start = process_time()
                # Solver initialization
                ap = hgs.AlgorithmParameters(timeLimit=time_limit,)  # seconds
                hgs_solver = hgs.Solver(parameters=ap,verbose=False)

                pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

                data = Routing.HyGeSe.generate_HyGeSe_data(inst_gen, requirements)

                # Save the original stdout
                result = hgs_solver.solve_cvrp(data)

                routes = Routing.HyGeSe.translate_routes(inst_gen, requirements,result.routes)

                return routes, result.cost, process_time() - start


            # Generate data dict for HyGeSe algorithm
            def generate_HyGeSe_data(inst_gen:instance_generator, requirements:dict) -> dict:
                data = dict()

                # data['distance_matrix'] = [[inst_gen.c[i,j] if i!=j else 0 for j in inst_gen.N] for i in inst_gen.N]
                data['distance_matrix'] = [[inst_gen.c[i,j] if i!=j else 0 for j in inst_gen.V if (j in requirements.keys() or j==0)] for i in inst_gen.V if (i in requirements.keys() or i==0)]
                data['demands'] = np.array([0] + list(requirements.values()))
                data['vehicle_capacity'] = inst_gen.Q
                data['num_vehicles'] = inst_gen.F
                data['depot'] = 0
            
                return data

            # Translate routes to environment suppliers
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
        def MixedIntegerProgram(purchase:dict[float],inst_gen:instance_generator,t:int):
            start = process_time()
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)
    
            N, V, A, distances, requirements = Routing.network_aux_methods.generate_complete_graph(inst_gen, pending_sup, requirements)
            A.append((0,inst_gen.M+1))
            distances[0,inst_gen.M+1] = 0

            model = Routing.MIP.generate_complete_MIP(inst_gen, N, V, A, distances, requirements)

            model.update()
            model.setParam('OutputFlag',0)
            model.setParam('MIPGap',0.1)
            model.optimize()

            routes, distances, loads = Routing.MIP.get_MIP_decisions(inst_gen, model, V, A, distances, requirements)
            cost = model.getObjective().getValue()

            return routes, distances, loads, process_time() - start    
        
        class MIP():
            # Generate complete MIP model
            def generate_complete_MIP(inst_gen:instance_generator, N:list, V:range, A:list, distances:dict, requirements:dict) -> gu.Model:
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
                    # model.addConstr(gu.quicksum(distances[i,j]*x[i,j,f] for (i,j) in A) <= inst_gen.d_max)

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
            def get_MIP_decisions(inst_gen:instance_generator, model:gu.Model, V:list, A:list, distances:dict, requirements:dict):
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
                            routes.append(route)
                            break
                    
                    dist.append(distance)
                    loads.append(load)
                    # print(f'Route {f} - {route}')


                return routes, dist, loads
        

        ''' Column generation algorithm '''
        def ColumnGeneration(purchase:dict[float], inst_gen:instance_generator,t:int):
            pending_sup, requirements = Routing.consolidate_purchase(purchase,inst_gen,t)

            N, V, A, distances, requirements = Routing.network_aux_methods.generate_complete_graph(inst_gen,pending_sup,requirements)

            master = Routing.Column_Generation.MasterProblem()
            modelMP, theta, RouteLimitCtr, NodeCtr = master.buidModel(inst_gen, N, distances)

            card_omega = len(theta)

            print("Entering Column Generation Algorithm",flush = True)
            while True:
                print('Solving Master Problem (MP)...', flush = True)
                modelMP.optimize()
                print('Value of LP relaxation of MP: ', modelMP.getObjective().getValue(), flush = True)

                # for j in range(card_omewga): #Retrieving solution of RMP
                #     if(theta[j].x!=0):
                #         print(f'theta({j}) = {theta[j].x}', flush = True)
                
                #Retrieving duals of master problem
                # lambdas = list()
                # lambdas.append(modelMP.getAttr("Pi", modelMP.getConstrs()[0])[0])
                # for i in N:
                #     lambdas += [modelMP.getAttr("Pi", modelMP.getConstrs()[i])]
                # lambdas = list(modelMP.getAttr("Pi", modelMP.getConstrs()))
                lambdas = []
                lambdas.append(modelMP.getAttr("Pi", RouteLimitCtr)[0])
                lambdas+= modelMP.getAttr("Pi", NodeCtr)

                # for i in range(len(lambdas)):
                #     print('lambda(',i,')=', lambdas[i], sep ='', flush = True)  

                # Solve subproblem (passing dual variables)
                print('Solving subproblem (AP):', card_omega, flush = True)
                
                a_star = dict()
                a_star.update({i:0 for i in N})
                shortest_path, a_star = Routing.Column_Generation.SubProblem.solveAPModel(lambdas, a_star, inst_gen, N, V, A, distances, requirements)
                minReducedCost = shortest_path[0]
                c_k = shortest_path[1]

                # Check termination condition
                if  minReducedCost >= -0.0005:
                    print("Column generation stops! \n", flush = True)
                    break
                else:
                    print('Minimal reduced cost (via CSP):', minReducedCost, '<0.', flush = True)
                    print('Adding column...', flush = True)
                    a_star = list(a_star.values())
                    a_star.append(1) #We add the 1 of the number of routes restrictions

                    newCol = gu.Column(a_star, modelMP.getConstrs())
                    theta.append(modelMP.addVar(vtype=gu.GRB.CONTINUOUS,obj=c_k,lb=0,column=newCol,name=f"theta_{card_omega}"))
                    card_omega+=1
                    # Update master model
                    modelMP.update()


            for v in modelMP.getVars():
                if v.x>0.5:
                    print('%s=%g' % (v.varName, v.x))
            
            for v in modelMP.getVars():
                v.setAttr("Vtype", gu.GRB.INTEGER)

            modelMP.optimize()

            print('(Heuristic) integer master problem:')
            print('Route time: %g' % modelMP.objVal)
            for v in modelMP.getVars():
                if v.x > 0.5:
                    print('%s %g' % (v.varName, v.x))
            
            print('Normal termination. -o-')
        
        class Column_Generation():
            # Master problem
            class MasterProblem:
                
                def buidModel(self, inst_gen:instance_generator, N:list, distances:dict, name:str = 'MasterProblem'):
                    modelMP = gu.Model(name)

                    modelMP.Params.Presolve = 0
                    modelMP.Params.Cuts = 0
                    modelMP.Params.OutputFlag = 0

                    modelMP, theta = self.generateVariables(inst_gen, modelMP, N, distances)
                    modelMP, RouteLimitCtr, NodeCtr = self.generateConstraints(inst_gen, modelMP, N, theta)
                    modelMP = self.generateObjective(modelMP)
                    modelMP.update()

                    return modelMP, theta, RouteLimitCtr, NodeCtr
            

                def generateVariables(self, inst_gen:instance_generator, modelMP:gu.Model, N:list, distances):
                    theta = list()
                    
                    #Initial set-covering model  (DUMMY INITIALIZATION)
                    def calculateDummyCost():
                        c_0 = 0
                        for i in N:
                            c_0+=2*distances[i,inst_gen.M+1]
                        return c_0

                    dummyCost = calculateDummyCost()
                    theta.append(modelMP.addVar(vtype = gu.GRB.CONTINUOUS, obj = dummyCost, lb = 0, name = "theta_0"))

                    for i in N:
                        route_cost = distances[0,i] + distances[i,inst_gen.M+1]
                        theta.append(modelMP.addVar(vtype=gu.GRB.CONTINUOUS,obj=route_cost,lb=0,name=f"theta_{i}"))

                    return modelMP, theta


                def generateConstraints(self, inst_gen:instance_generator, modelMP:gu.Model, N:list, theta:list):
                    NodeCtr = list()                #Node covering constraints
                    for i in N:
                        NodeCtr.append(modelMP.addConstr(theta[i]>=1, f"Set_Covering_{i}")) #Set covering constraints
                    
                    RouteLimitCtr = list()          #Limits the number of routes
                    RouteLimitCtr.append(modelMP.addConstr(gu.quicksum(theta[i] for i in range(len(theta))) <= inst_gen.F, 'Route_Limit_Ctr')) #Routes limit Constraints

                    return modelMP, RouteLimitCtr, NodeCtr


                def generateObjective(self, modelMP:gu.Model):
                    modelMP.modelSense = gu.GRB.MINIMIZE
                    return modelMP 

            # Auxiliary problem
            class SubProblem:

                def solveAPModel(lambdas, a_star, inst_gen, N, V, A, distances, requirements):
                    
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
                    modelAP.addConstr(gu.quicksum(x[0,j] for j in N) == 1, "Depart from depot")

                    # 4. All vehicles arrive at depot
                    modelAP.addConstr(gu.quicksum(x[i,inst_gen.M+1] for i in N) == 1, "Reach the depot")

                    # 5. Flow preservation
                    for i in N:
                        modelAP.addConstr(gu.quicksum(x[i,j] for j in V if (i,j) in A) - gu.quicksum(x[j,i] for j in V if (j,i) in A) == 0, f'Flow conservation_{i}')

                    # 6. Max distance per vehicle
                    modelAP.addConstr(gu.quicksum(distances[i,j]*x[i,j] for (i,j) in A) <= inst_gen.d_max, 'Max distance')

                    # 7. Max capacity per vehicle
                    modelAP.addConstr(gu.quicksum(requirements[i]*gu.quicksum(x[i,j] for j in V if (i,j) in A) for i in N) <= inst_gen.Q, "Capacity")

                    # 8. Distance tracking/No loops
                    for (i,j) in A:
                        modelAP.addConstr(w[i]+distances[i,j]-w[j] <= (1-x[i,j])*1e9, f'Distance tracking_{i}{j}')

                    #Shortest path objective
                    c_trans = dict()
                    for (i,j) in A:
                        if i in N:
                            c_trans[i,j] = distances[i,j]-lambdas[i]
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
                    return [modelAP.objVal, route_cost], a_star


        ''' Auxiliary methods for network management '''
        class network_aux_methods():
            # Generate vertices and arches for a complete graph
            def generate_complete_graph(inst_gen: instance_generator, pending_sup:list, requirements:dict) -> tuple[list,list,list]:
                N = pending_sup
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
        def consolidate_purchase(purchase,inst_gen,t) -> tuple[list,dict]:
            # purchse is given for suppliers and products
            if type(list(purchase.keys())[0]) == tuple:
                pending_suppliers = list()
                requirements = dict()
                for i in inst_gen.Suppliers:
                    req = sum(purchase[i,k] for k in inst_gen.K_it[i,t] if (i,k) in purchase.keys())
                    if req > 0:
                        pending_suppliers.append(i)
                        requirements[i] = req

                return pending_suppliers, requirements
            # purchase is given for products
            else:
                return list(purchase.keys()), purchase
