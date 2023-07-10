"""
@author: Juan Betancourt
################################## Modules ##################################
"""
### SC classes
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP

### Basic Librarires
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
from copy import deepcopy


class RoutingV():
    # Sumarizes and compares routing solutions
    def compare_routing_strategies(inst_gen:instance_generator, data:dict[list]):
        print('Policy \t#Veh \tDist \tAvgUt \tavgEff \tTime \tRealCost')
        for strategy, performance in data.items():
            if strategy in ['MIP','CG'] :
                num_routes = len([i for i in performance[0] if i != [0,0]])
                avg_ut = round(sum(performance[2])/(num_routes*inst_gen.Q), 2)
                avg_eff = round(sum(performance[1])/(num_routes*inst_gen.d_max), 2)
                print(f'{strategy} \t{len([i for i in performance[0] if i != [0,0]])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(performance[3],2)} \t{round(performance[4],2)}')
            elif strategy == 'HyGeSe':
                print(f'{strategy} \t{len(performance[0])} \t{round(performance[1],2)} \t- \t- \t{round(performance[2],2)} \t{round(performance[3],2)}')
            else:
                avg_ut = round(sum(performance[2])/(len(performance[0]*inst_gen.Q)), 2)
                avg_eff = round(sum(performance[1])/len(performance[0]*inst_gen.d_max), 2)
                print(f'{strategy} \t{len(performance[0])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(performance[3],2)} \t{round(performance[4],2)}')
                
        # print('Policy \t#Veh \tDist \tAvgUt \tavgEff \tRealCost')
        # for strategy, performance in data.items():
        #     if strategy in ['MIP','CG'] :
        #         num_routes = len([i for i in performance[0] if i != [0,0]])
        #         avg_ut = round(sum(performance[2])/(num_routes*inst_gen.Q), 2)
        #         avg_eff = round(sum(performance[1])/(num_routes*inst_gen.d_max), 2)
        #         print(f'{strategy} \t{len([i for i in performance[0] if i != [0,0]])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(performance[4],2)}')
        #     elif strategy == 'HyGeSe':
        #         print(f'{strategy} \t{len(performance[0])} \t{round(performance[1],2)} \t- \t-  \t{round(performance[3],2)}')
        #     else:
        #         avg_ut = round(sum(performance[2])/(len(performance[0]*inst_gen.Q)), 2)
        #         avg_eff = round(sum(performance[1])/len(performance[0]*inst_gen.d_max), 2)
        #         print(f'{strategy} \t{len(performance[0])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(performance[4],2)}')


    # Plot routes
    def render_routes(inst_gen:instance_generator,routes:list,save:bool=False):
        G = nx.MultiDiGraph()

        # Nodes
        node_list = [0]
        node_list += inst_gen.Suppliers
        G.add_nodes_from(node_list)

        node_color = ['green']
        node_color += ['tab:purple' for i in inst_gen.Suppliers]
        nodes_to_draw = deepcopy(node_list)  

        # Edges
        edges = []
        edge_colors = []
        orders = {}
        for i in range(len(routes)):
            route = routes[i]
            for node in range(len(route) - 1):
                edge = (route[node], route[node + 1])
                edges.append(edge)
                orders[edge] = i

        G.add_edges_from(edges) 
        edges = G.edges()
        colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                       'tab:pink', 'tab:grey', 
                       'black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                        'tab:pink', 'tab:grey']
        for edge in edges:
            color = colors[orders[edge]]
            edge_colors.append(color)

        # pos = {c: (self.C[c]['x'], self.C[c]['y']) for c in self.Costumers}
        # pos.update({s: (self.S[s]['x'], self.S[s]['y']) for s in self.Stations})
        # pos['D'] = (self.D['x'], self.D['y'])

        nx.draw_networkx(G,pos=inst_gen.coor, with_labels = True, nodelist = nodes_to_draw, 
                         node_color = node_color, edge_color = edge_colors, alpha = 0.8, 
                         font_size = 7, node_size = 200)
        if save:
            plt.savefig(inst_gen.path + 'routes.png', dpi = 600)
        plt.show()


    # Plot routes of various routing strategies to compare
    def render_routes_diff_strategies(inst_gen:instance_generator,solutions:list,save:bool=False):
        G = nx.MultiDiGraph()

        # Nodes
        node_list = [0]
        node_list += inst_gen.Suppliers
        G.add_nodes_from(node_list)

        node_color = ['green']
        node_color += ['tab:purple' for i in inst_gen.Suppliers]
        nodes_to_draw = deepcopy(node_list)

        # Edges
        edges = []
        edge_colors = []
        orders = {}
        for idx, routes in enumerate(solutions):
            for i in range(len(routes)):
                route = routes[i]
                for node in range(len(route) - 1):
                    edge = (route[node], route[node + 1])
                    edges.append(edge)
                    orders[edge] = idx
            
        G.add_edges_from(edges) 
        edges = G.edges()
        colors = ['black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                       'tab:pink', 'tab:grey', 
                       'black', 'red', 'green', 'blue', 'purple', 'orange', 'pink', 'grey', 
                       'yellow', 'tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange', 
                        'tab:pink', 'tab:grey']
        for edge in edges:
            color = colors[orders[edge]]
            edge_colors.append(color)

        nx.draw_networkx(G,pos=inst_gen.coor, with_labels = True, nodelist = nodes_to_draw, 
                         node_color = node_color, edge_color = edge_colors, alpha = 0.8, 
                         font_size = 7, node_size = 200)
        if save:
            plt.savefig(inst_gen.path + 'routing_strategies_comparison.png', dpi = 600)
        plt.show()


    # Scatter plot of solutions (transport cost vs. Purchasing delta)
    def plot_solutions(inst_gen:instance_generator,routing_performance:dict):
        x = dict()
        y = dict()
        for strategy in routing_performance[1].keys():
            x[strategy] = list()
            y[strategy] = list()
        
        for ep,data in list(routing_performance.items()):
            for strategy in x.keys():
                if strategy != 'HyGeSe':
                    x[strategy].append(routing_performance[ep][strategy][4])
                    y[strategy].append(routing_performance[ep][strategy][4])
                else:
                    x[strategy].append(routing_performance[ep][strategy][3])
                    y[strategy].append(routing_performance[ep][strategy][3])

        # Set up figure and axes
        fig, ax = plt.subplots()

        # Plot scatter plots for each series
        for series_name in x.keys():
            ax.scatter(x[series_name], y[series_name], label=series_name)

        # Add labels and title
        ax.set_xlabel('Transport cost')
        ax.set_ylabel('Purchase delta')
        ax.set_title('Routing strategies')

        # Add legend
        ax.legend()

        # Set aspect ratio to equal
        ax.set_aspect('equal')

        # Gridlines
        ax.grid(True, linestyle='--')

        # Show plot
        plt.show()


    # Displays the historic availability of a given route for a given product
    def route_availability_per_product(route:list, product:int, inst_gen:instance_generator, env:steroid_IRP, include_ceros:bool = False):
        series = list()
        labels = list()
        for i in route[1:-1]:
            if include_ceros:   series.append(inst_gen.hist_q[env.t][i,product])
            else:               series.append([ii for ii in inst_gen.hist_q[env.t][i,product] if ii != 0])
            
            labels.append(str(i))

        avg = RoutingV.return_mean([serie[j] for serie in series for j in range(len(serie))])
        bracket = RoutingV.return_brackets([serie[j] for serie in series for j in range(len(serie))], avg)

        plt.hist(series, density = True, histtype = 'bar', label = labels)
        
        plt.axvline(x = avg, ymin = 0, ymax = 0.9, color = 'black', label = 'Mean')
        plt.axvline(x = bracket[0], ymin = 0, ymax = 0.8, color = 'black', linestyle = ':', linewidth = 0.85)
        plt.axvline(x = bracket[1], ymin = 0, ymax = 0.8, color = 'black', linestyle = ':', linewidth = 0.85)

        plt.legend(title = 'Suppliers', prop={'size':10})
        plt.title(f'Availability of product {product}')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')

        plt.show()
    

    # Displays the historic total availability of the suppiers of a given route
    def route_total_availability(route:list, inst_gen:instance_generator, env:steroid_IRP, include_ceros:bool = False):
        series = list()
        labels = list()
        for i in route[1:-1]:
            vals = list()
            for k in inst_gen.Products:
                if include_ceros:
                    vals.extend(inst_gen.hist_q[env.t][i,k])    
                else:
                    vals.extend([ii for ii in inst_gen.hist_q[env.t][i,k] if ii != 0])

            series.append(vals)
            labels.append(str(i))
        
        avg = Routing_Visualizations.return_mean([serie[j] for serie in series for j in range(len(serie))])
        bracket = Routing_Visualizations.return_brackets([serie[j] for serie in series for j in range(len(serie))], avg)
        
        plt.hist(series, density = True, histtype = 'bar', label = labels)

        plt.axvline(x = avg, ymin = 0, ymax = 0.9, color = 'black', label = 'Mean')
        plt.axvline(x = bracket[0], ymin = 0, ymax = 0.8, color = 'black', linestyle = ':', linewidth = 0.85)
        plt.axvline(x = bracket[1], ymin = 0, ymax = 0.8, color = 'black', linestyle = ':', linewidth = 0.85)

        plt.legend(title = 'Suppliers', prop={'size':10})
        plt.title(f'Total Availability')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()
    

    # Displays the historic avaiability of different routes for a given product
    def routes_availability_per_product(routes:list, product:int, inst_gen:instance_generator, env:steroid_IRP, include_ceros:bool = False):
        series = list()
        labels = list()
        avgs = list()
        bracks = list()
        cols = mcolors.TABLEAU_COLORS
        # colors = [cols[key] for i, key in enumerate(list(cols.keys())) if i < len(routes)]
        colors = list()

        for i, route in enumerate(routes):
            if include_ceros:
                series.append([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]])
            else:
                series.append([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product] if j != 0])

            labels.append(str(i))
            avgs.append(Routing_Visualizations.return_mean([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]]))
            bracks.append(Routing_Visualizations.return_brackets([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]],avgs[-1]))
            colors.append(list(cols.values())[i])

        # bracket = Routing_Visualizations.return_brackets([serie[j] for serie in series for j in range(len(series))], avg)
        
        plt.hist(series, density = True, histtype = 'bar', color = colors, label = labels)
        
        for i, route in enumerate(routes):
            plt.axvline(x = avgs[i], ymin = 0, ymax = 0.975, color = colors[i])
            plt.axvline(x = bracks[i][0], ymin = 0, ymax = 0.9, color = colors[i], linestyle = ':', linewidth = 1.5)
            plt.axvline(x = bracks[i][1], ymin = 0, ymax = 0.9, color = colors[i], linestyle = ':', linewidth = 1.5)

        plt.legend(title = 'Routes', prop={'size':10})
        plt.title(f'Availability of product {product}')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()
    

    # Displays the historic total avaiability of different routes
    def routes_total_availability(routes:list, inst_gen:instance_generator, env:steroid_IRP, include_ceros:bool = False):
        series = list()
        labels = list()
        avgs = list()
        bracks = list()
        cols = mcolors.TABLEAU_COLORS
        # colors = [cols[key] for i, key in enumerate(list(cols.keys())) if i < len(routes)]
        colors = list()

        for i, route in enumerate(routes):
            vals = list()
            for j in route[1:-1]:
                for k in inst_gen.Products:
                    if include_ceros:
                        vals.extend(inst_gen.hist_q[env.t][j,k]) 
                    else:
                        vals.extend([ii for ii in inst_gen.hist_q[env.t][j,k] if ii != 0])

            series.append(vals) 
            labels.append(str(i))
            avgs.append(Routing_Visualizations.return_mean(vals))
            bracks.append(Routing_Visualizations.return_brackets(vals, avgs[-1]))
            colors.append(list(cols.values())[i])



        plt.hist(series, density = True, histtype = 'bar', label = labels)

        for i, route in enumerate(routes):
            plt.axvline(x = avgs[i], ymin = 0, ymax = 0.975, color = colors[i])
            plt.axvline(x = bracks[i][0], ymin = 0, ymax = 0.9, color = colors[i], linestyle = ':', linewidth = 1.5)
            plt.axvline(x = bracks[i][1], ymin = 0, ymax = 0.9, color = colors[i], linestyle = ':', linewidth = 1.5)

        plt.legend(title = 'Routes', prop={'size':10})
        plt.title(f'Total Availability')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()


    # Auxiliary function to return mean 
    def return_mean(values:list):
        return (sum(values)/len(values))


    # Auxiliary function to return limits of std interval
    def return_brackets(values:list, mean:float):
        return mean + np.std(values), mean - np.std(values)


class InventoryV():
    pass


class PerishabilityV():
    pass