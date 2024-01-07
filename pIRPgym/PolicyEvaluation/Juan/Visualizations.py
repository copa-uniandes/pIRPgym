"""
@author: Juan Betancourt
################################## Modules ##################################
"""
### Basic Librarires
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import networkx as nx
from copy import deepcopy

import seaborn as sns
from sklearn.neighbors import KernelDensity

### SC classes
from ...Blocks.InstanceGenerator import instance_generator
from ...Blocks.pIRPenv import steroid_IRP


class RoutingV():
    colors = ['purple', 'red', 'blue', 'green', 'orange', 'cyan', 'magenta','black','brown']\
        +['purple', 'red', 'blue', 'green', 'orange', 'cyan', 'magenta','black','brown']\
        +['purple', 'red', 'blue', 'green', 'orange', 'cyan', 'magenta','black','brown']\
        +['purple', 'red', 'blue', 'green', 'orange', 'cyan', 'magenta','black','brown']

    # Sumarizes and compares routing solutions
    @staticmethod
    def DEP_compare_routing_strategies(inst_gen:instance_generator,data:dict):
        print('Policy \t#Veh \tDist \tAvgUt \tavgEff \tTime \tRealCost')
        for strategy, performance in data.items():
            if strategy in ['MIP','CG'] :
                num_routes = len([i for i in performance[0] if i != [0,0]])
                avg_ut = round(sum(performance[2])/(num_routes*inst_gen.Q), 2)
                avg_eff = round(sum(performance[1])/(num_routes*inst_gen.d_max), 2)
                rtime = performance[3]
                if rtime <1000:
                    print(f'{strategy} \t{len([i for i in performance[0] if i != [0,0]])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(rtime,2)} \t{round(performance[4],2)}')
                elif rtime <9999:
                    print(f'{strategy} \t{len([i for i in performance[0] if i != [0,0]])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(rtime,1)} \t{round(performance[4],2)}')
                else:
                    print(f'{strategy} \t{len([i for i in performance[0] if i != [0,0]])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(rtime)} \t{round(performance[4],2)}')
            elif strategy == 'HyGeSe':
                print(f'{strategy} \t{len(performance[0])} \t{round(performance[1],2)} \t- \t- \t{round(performance[2],2)} \t{round(performance[3],2)}')
            else:
                avg_ut = round(sum(performance[2])/(len(performance[0]*inst_gen.Q)), 2)
                avg_eff = round(sum(performance[1])/len(performance[0]*inst_gen.d_max), 2)
                print(f'{strategy} \t{len(performance[0])} \t{round(sum(performance[1]))} \t{avg_ut} \t{avg_eff} \t{round(performance[3],2)} \t{round(performance[4],2)}')
        

    # Plot routes
    @staticmethod
    def render_routes(inst_gen:instance_generator,routes:list,save_fig:bool=False):
        G = nx.MultiDiGraph()

        # Nodes
        node_list = [0]
        node_list += inst_gen.Suppliers
        G.add_nodes_from(node_list)

        node_color = ['green']
        node_color += ['tab:red' for i in inst_gen.Suppliers]
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
        colors = ['red','orange','blue','green','tab:purple']
        for edge in edges:
            color = colors[orders[edge]]
            edge_colors.append('blue')

        # pos = {c: (self.C[c]['x'], self.C[c]['y']) for c in self.Costumers}
        # pos.update({s: (self.S[s]['x'], self.S[s]['y']) for s in self.Stations})
        # pos['D'] = (self.D['x'], self.D['y'])

        # nx.draw_networkx(G,pos=inst_gen.coor, with_labels = True, nodelist = nodes_to_draw, 
        #                  node_color = node_color, edge_color = edge_colors, alpha = 0.8, 
        #                  font_size = 7, node_size = 200)
        

        nx.draw_networkx(   G,
                            pos=inst_gen.coor,
                            with_labels=True,
                            nodelist=nodes_to_draw,
                            node_color=node_color,
                            edge_color=edge_colors,
                            alpha=0.8,
                            font_size=8,  # Adjust font size
                            font_color='black',  # Font color for node labels
                            font_weight='bold',  # Font weight for node labels
                            arrowsize=10,  # Size of arrowheads for directed edges
                            width=1.5,  # Width of edges
                            edgecolors='black',  # Color of node borders
                            node_shape='o',  # Node shape (e.g., 'o' for circles)
                            font_family='arial',  # Font family for node labels
                        )
        
        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add ticks to the x (bottom) and y (left) axis
        ax.tick_params(axis='both', which='both', bottom=True, left=True,labelsize=100)
        
        
        ax.set_xticks([0, 200, 400, 600, 800, 1000])
        ax.set_yticks([0, 200, 400, 600, 800, 1000])

        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a title to the graph
        plt.title("Flower 43", fontsize=12)
        
        if save_fig:
            plt.savefig('./SuppliersLayout.png', dpi = 400)
        plt.show()


    @staticmethod
    def plot_metric(routing_performance,metric,x_axis:str='Time step',x_values=None):
        """
        Plot the evolution of a specific metric for different routing policies.

        Parameters:
        - routing_performance (dict): Dictionary containing routing policies and their metrics.
        - metric (str): Indicator to plot ('Obj', 'time', 'vehicles', 'reactive_missing', 'extra_cost').
        """
        # Set up figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define a list of colors and markers for better visibility
        colors = RoutingV.colors
        markers = ['o', 's', '^', 'D', '*', 'v', 'p', 'H', 'X']
        translator = {}

        # Plot the evolution for each routing policy
        for i, (policy, data) in enumerate(routing_performance.items()):
            if metric in data:
                if x_values == None: 
                    ax.plot(data[metric],label=f'{policy}',color=colors[i % len(colors)],marker=markers[i % len(markers)],
                            linestyle='-',markersize=4,linewidth=1)
                else: 
                    ax.plot(x_values,data[metric],label=f'{policy}',color=colors[i % len(colors)],marker=markers[i % len(markers)],
                            linestyle='-',markersize=4,linewidth=1)

        # Add labels and a legend
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel(x_axis, fontsize=12)
        
        # ax.set_ylabel(metric, fontsize=12)
        # ax.set_title('Effectiveness of Initialization Configurations', fontsize=14)
        # ax.legend(fontsize=10, loc='upper right')

        ax.set_ylabel('gap',fontsize=12)
        ax.set_title('Gap of Alpha Configurations', fontsize=14)
        legend_labels = ['NN','CG','0.1','0.2','0.4','0.6','0.8']
        ax.legend(legend_labels,fontsize=10,loc='upper right')
        

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.savefig('./metric.png', dpi = 400)

        # Show the plot
        plt.show()

    
    @staticmethod
    def plot_supplier_distributions(bounds,save_fig:bool=False):
        """
        Plot boxplots with intervals for suppliers.

        Parameters:
        - supplier_intervals (dict): Dictionary with supplier intervals.

        Returns:
        None
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract supplier numbers and intervals
        suppliers = [int(s) for s in bounds.keys()]
        intervals = [interval for interval in bounds.values()]

        # Plot brackets for each supplier interval
        for i, (supplier, interval) in enumerate(zip(suppliers, intervals)):
            lower, upper = interval
            ax.plot([i, i], [lower, upper], color=RoutingV.colors[i], linewidth=2)
            ax.plot([i - 0.2, i + 0.2], [lower, lower], color=RoutingV.colors[i], linewidth=2)
            ax.plot([i - 0.2, i + 0.2], [upper, upper], color=RoutingV.colors[i], linewidth=2)

        # Set x-axis labels to supplier numbers
        ax.set_xticks(range(len(suppliers)))
        ax.set_xticklabels(suppliers)

        
        # Set labels and title
        ax.set_xlabel('Supplier')
        ax.set_ylabel('Units')
        ax.set_title("Suppliers' Availability Distribution")

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        if save_fig:
            plt.savefig('./SuppliersDistribution',dpi=400)
        else:            
            plt.show()

        


    @staticmethod
    def n_plot(n_table:list,bin_cod:list,suppliers:tuple):
        """
        Generate a bar plot for service levels across different solutions.

        Parameters:
        - solution_heights (list): Heights of bars for each solution.

        Returns:
        None
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bar plot for each solution
        for i, height in enumerate(n_table):
            if RoutingV._contains_subset(bin_cod[i],suppliers):
                sns.barplot(x=[i], y=[height], label=f'Flower {i}', alpha=0.7, ax=ax)

        # Set labels and title
        ax.set_xlabel('Flowers')
        ax.set_ylabel('Number of observations')
        ax.set_title('Recurrence of Flowers')

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(title='Flowers', loc='upper left', bbox_to_anchor=(1, 1))

        plt.savefig('./n_table.png', dpi = 600)

        plt.show() 


    @staticmethod
    def _contains_subset(binary_encoding,supplier_list):
        """
        Check if a binary encoding contains all the suppliers from a given list.

        Parameters:
        - binary_encoding (numpy array): Binary encoding of suppliers.
        - supplier_list (list): List of suppliers to check.

        Returns:
        bool: True if all 1s in the binary encoding are in the supplier list, False otherwise.
        """
        if supplier_list[0]=='All': return True
        flag = True
        for i,pos in enumerate(supplier_list):
            if binary_encoding[pos-1]==0:
                flag = False
                break
        return flag


    @staticmethod
    def plot_indicators(metrics,bincod,indicator1,indicator2,suppliers):
        """
        Plot a scatter plot for two service level types across different solutions.

        Parameters:
        - solution_data (list of tuples): Each tuple contains the service level values for both types in a solution.

        Returns:
        None
        """
        pos1,pos2 = RoutingV._get_indicator_positions(indicator1,indicator2)

        solution_data = [(obs[pos1],obs[pos2]) for obs in metrics]
        fig,ax = plt.subplots(figsize=(10, 6))

        # Create a DataFrame for seaborn plotting
        df = pd.DataFrame(solution_data,columns=[indicator1,indicator2])

        # Plot the scatter plot with consistent colors for each solution
        for i, solution in enumerate(solution_data):
            if RoutingV._contains_subset(bincod[i],suppliers):
                sns.scatterplot(x=indicator1, y=indicator2, data=pd.DataFrame([solution], columns=df.columns),
                            label=f'Flower {i}', s=150, ax=ax)

        # Set labels and title
        ax.set_xlabel(indicator1)
        ax.set_ylabel(indicator2)
        ax.set_title('Performance Indicators Across Flowers')

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(title='Flowers', loc='upper left', bbox_to_anchor=(1, 1))

        plt.show()

    
    @staticmethod
    def _get_indicator_positions(indicator1,indicator2):
        positions = {'Fixed Service Level':2,'Dynamic Service Level':3,'Cost per Supplier':1,'Price Delta':4}
        return positions[indicator1],positions[indicator2]


    @staticmethod
    def plot_supplier_availabilities(supplier_distributions):
        """
        Plot box plots for supplier distributions.

        Parameters:
        - supplier_distributions (dict): Dictionary with supplier distributions.

        Returns:
        None
        """
        supplier_distributions = {str(key):value for key,value in supplier_distributions.items() if sum(value)>0}

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract supplier numbers and distributions
        suppliers = [int(s) for s in supplier_distributions.keys()]
        distributions = [distribution for distribution in supplier_distributions.values()]

        # Plot box plots for each supplier
        boxprops = dict(linewidth=2, color='black')
        whiskerprops = dict(linewidth=2, linestyle='--', color='black')
        capprops = dict(linewidth=2, color='black')
        medianprops = dict(linewidth=2, linestyle='-', color='red')
        bp = ax.boxplot(distributions, patch_artist=True, boxprops=boxprops,
                        whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)

        # Custom fill colors for boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # Set x-axis labels to supplier numbers
        ax.set_xticklabels(suppliers)

        # Set labels and title
        ax.set_xlabel('Supplier')
        ax.set_ylabel('Distribution')
        ax.set_title('Supplier Distributions')

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()


    @staticmethod
    def count_generated(generator):
        CG_num = 0
        GA_num = 0
        both_num = 0
        for i,gens in enumerate(generator):
            if 'CG' in gens:    CG_num+=1
            if 'GA' in gens:    GA_num+=1 
            if 'CG' in gens and 'GA' in gens:   both_num += 1
        return CG_num,GA_num,both_num


    @staticmethod
    def plot_flower_comparison(historics,indicator1,indicator2,flower1,flower2):
        """
        Plot the observations of two flowers with mean coordinates highlighted.

        Parameters:
        - flower1 (tuple): Tuple containing x and y coordinates of the first flower.
        - flower2 (tuple): Tuple containing x and y coordinates of the second flower.

        Returns:
        None
        """
        pos1,pos2 = RoutingV._get_historics_positions(indicator1,indicator2)
        flower1 = (historics[flower1][pos1],historics[flower1][pos2])
        flower2 = (historics[flower2][pos1],historics[flower2][pos2])


        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot observations of the first flower in blue
        sns.scatterplot(x=flower1[0], y=flower1[1], label='Flower 1', marker='.',color='purple', s=50, alpha=0.7, ax=ax)

        # Plot observations of the second flower in red
        sns.scatterplot(x=flower2[0], y=flower2[1], label='Flower 2', marker='.',color='red', s=50, alpha=0.7, ax=ax)

        # Calculate mean coordinates for each flower
        mean_flower1 = (np.mean(flower1[0]), np.mean(flower1[1]))
        mean_flower2 = (np.mean(flower2[0]), np.mean(flower2[1]))

        # Plot mean coordinates with a bigger marker
        sns.scatterplot(x=[mean_flower1[0]], y=[mean_flower1[1]],
                        marker='o', s=150, color='purple')#, label='F1 Avg. Performance')
        sns.scatterplot(x=[mean_flower2[0]], y=[mean_flower2[1]],
                        marker='o', s=150, color='red')#, label='F2 Avg. Performance')

        # Set labels and title
        ax.set_xlabel(indicator1)
        ax.set_ylabel(indicator2)
        ax.set_title('Flower Comparison')

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    @staticmethod
    def _get_historics_positions(indicator1,indicator2):
        positions = {'Fixed Service Level':0,'Dynamic Service Level':1,'Price Delta':2}
        return positions[indicator1],positions[indicator2]


    # Plot routes of various routing strategies (comparison)
    @staticmethod
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
    @staticmethod
    def plot_solutions(inst_gen:instance_generator,routing_performance:dict):
        x = dict()
        y = dict()

        markers = ['.','p','x','*','^','P','s']
        cols = ['tab:red','tab:orange','tab:blue','tab:green','tab:purple']

        for strategy in routing_performance[1].keys():
            x[strategy] = list()
            y[strategy] = list()
        
        for ep,data in list(routing_performance.items()):
            for strategy in x.keys():
                if strategy != 'HyGeSe':
                    x[strategy].append(sum(data[strategy][1]))
                    y[strategy].append(data[strategy][4])
                else:
                    x[strategy].append(data[strategy][1])
                    y[strategy].append(data[strategy][3])

        # Set up figure and axes
        fig, ax = plt.subplots()

        # Plot scatter plots for each series
        for ii,series_name in enumerate(list(x.keys())):
            for idx in range(len(list(routing_performance.keys()))):
                ax.scatter(x[series_name][idx],y[series_name][idx],color=cols[ii],marker=markers[idx])

        # Add labels and title
        ax.set_xlabel('Transport cost')
        ax.set_ylabel('Purchase delta')
        ax.set_title('Routing strategies')

        # Create legend handles for colors only
        legend_handles = []
        for ii, series_name in enumerate(list(x.keys())):
            legend_handles.append(mpatches.Patch(color=cols[ii], label=series_name))

        # Add legend with custom handles
        ax.legend(handles=legend_handles)

        # Gridlines
        ax.grid(True, linestyle='--')

        # Show plot
        plt.show()


    # Scatter plot of solutions (NORMALIZED transport cost vs. Purchasing delta)
    @staticmethod
    def plot_solutions_standarized(inst_gen:instance_generator,routing_performance:dict):
        x = dict()
        y = dict()
        for strategy in routing_performance[1].keys():
            x[strategy] = list()
            y[strategy] = list()

        for ep,data in list(routing_performance.items()):
            max_routing = 0
            max_purchase = 0
            for strategy in x.keys():
                if strategy != 'HyGeSe':
                    routing_c = sum(data[strategy][1]); purchase_c = data[strategy][4]
                    x[strategy].append(routing_c); y[strategy].append(purchase_c)
                    if routing_c > max_routing: max_routing = routing_c
                    if purchase_c > max_purchase: max_purchase = purchase_c
                else:
                    routing_c = data[strategy][1]; purchase_c = data[strategy][3]
                    x[strategy].append(routing_c); y[strategy].append(purchase_c)
                    if routing_c > max_routing: max_routing = routing_c
                    if purchase_c > max_purchase: max_purchase = purchase_c
            
            for strategy in x.keys():
                x[strategy][-1] /= max_routing
                y[strategy][-1] /= max_purchase

        # Set up figure and axes
        fig, ax = plt.subplots()

        # Plot scatter plots for each series
        markers = ['.','p','x','*','^','P','s']
        cols = ['tab:red','tab:orange','tab:blue','tab:green','tab:purple']
        # Plot scatter plots for each series
        for ii,series_name in enumerate(list(x.keys())):
            for idx in range(len(list(routing_performance.keys()))):
                ax.scatter(x[series_name][idx],y[series_name][idx],color=cols[ii],marker=markers[idx])

        # Add labels and title
        ax.set_xlabel('Transport cost')
        ax.set_ylabel('Purchase delta')
        ax.set_title('Routing strategies')

        # Create legend handles for colors only
        legend_handles = []
        for ii, series_name in enumerate(list(x.keys())):
            legend_handles.append(mpatches.Patch(color=cols[ii], label=series_name))

        # Add legend with custom handles
        ax.legend(handles=legend_handles)

        # Gridlines
        ax.grid(True, linestyle='--')

        # Show plot
        plt.show()


    # Plot solutions in vertical lines
    @staticmethod
    def plot_vertical_lines(inst_gen:instance_generator,routing_performance:dict,st:str,name:str,col:str):
        routing_costs = list()
        purchasing_delta = list()

        t_vals = list()
        p_vals = list()

        for ep,data in list(routing_performance.items()):
            ttt = list()
            ppp = list()

            max_routing = 0
            max_purchase = 0
            for strategy in data.keys():
                if strategy != 'HyGeSe':
                    routing_c = sum(data[strategy][1]); purchase_c = data[strategy][4]
                    ttt.append(routing_c); ppp.append(purchase_c)
                    if routing_c > max_routing: max_routing = routing_c
                    if purchase_c > max_purchase: max_purchase = purchase_c
                else:
                    routing_c = data[strategy][1]; purchase_c = data[strategy][3]
                    ttt.append(routing_c); ppp.append(purchase_c)
                    if routing_c > max_routing: max_routing = routing_c
                    if purchase_c > max_purchase: max_purchase = purchase_c
                
                if st==strategy:
                    t_vals.append(routing_c)
                    p_vals.append(purchase_c) 
            
            for i in range(len(ttt)):
                routing_costs.append(ttt[i]/max_routing)
                purchasing_delta.append(ppp[i]/max_purchase)

            t_vals[-1]/=max_routing
            p_vals[-1]/=max_purchase

        list1=routing_costs;list2=purchasing_delta; a=t_vals;b=p_vals

        # Calculate minimum and maximum values
        min_val1, max_val1 = min(list1), max(list1)
        min_val2, max_val2 = min(list2), max(list2)

        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Plot the values of list a
        ax.scatter([0.5]*len(a), a, color=col, label='List a',marker='D')

        # Plot the values of list b
        ax.scatter([1.5]*len(b), b, color=col, label='List b',marker='D')
        
        # Plot vertical lines for list1
        ax.vlines(0.5, min_val1, max_val1, color='black', linewidth=1)
        ax.hlines([min_val1, max_val1], 0.45, 0.55, colors='black', linewidth=2)

        # Plot vertical lines for list2
        ax.vlines(1.5, min_val2, max_val2, color='black', linewidth=1)
        ax.hlines([min_val2, max_val2], 1.45, 1.55, colors='black', linewidth=2)

        

        # Set the x-axis limits and labels
        ax.set_xlim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['Transport cost', 'Purchase delta'], fontsize=12)

        # Set the y-axis limits
        min_val = min(min_val1, min_val2)
        max_val = max(max_val1, max_val2)
        ax.set_ylim(min_val - 0.1*(max_val - min_val), max_val + 0.1*(max_val - min_val))

        # Add labels and title
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(f'Performance of {name}', fontsize=14)


        # Show the plot
        plt.show()


    class availability_display():

        # Displays the historic availability of a given route for a given product
        @staticmethod
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
        @staticmethod
        def route_total_availability(route:list,inst_gen:instance_generator,env:steroid_IRP,include_ceros:bool = False):
            series = list()
            labels = list()
            for i in route[1:-1]:
                vals = list()
                for k in inst_gen.K_it[i,0]:
                    if include_ceros:
                        vals.extend(inst_gen.hist_q[0][i,k])    
                    else:
                        vals.extend([ii for ii in inst_gen.hist_q[0][i,k] if ii != 0])

                series.append(vals)
                labels.append(str(i))
            
            avg = RoutingV.return_mean([serie[j] for serie in series for j in range(len(serie))])
            bracket = RoutingV.return_brackets([serie[j] for serie in series for j in range(len(serie))], avg)
            
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
        @staticmethod
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
                avgs.append(RoutingV.return_mean([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]]))
                bracks.append(RoutingV.return_brackets([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]],avgs[-1]))
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
        @staticmethod
        def routes_total_availability(routes:list,inst_gen:instance_generator,env:steroid_IRP,include_ceros:bool=False,title:str=''):
            series = list()
            labels = list()
            avgs = list()
            bracks = list()
            cols = mcolors.TABLEAU_COLORS
            # colors = [cols[key] for i, key in enumerate(list(cols.keys())) if i < len(routes)]
            cols = ['tab:purple','tab:red','tab:green','tab:brown']
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
                avgs.append(RoutingV.return_mean(vals))
                bracks.append(RoutingV.return_brackets(vals, avgs[-1]))
                colors.append(cols[i])


            plt.hist(series,density=True,histtype='bar',color=colors,label=labels)

            for i, route in enumerate(routes):
                plt.axvline(x=avgs[i],ymin=0,ymax=0.975,color=colors[i])
                plt.axvline(x=bracks[i][0],ymin=0,ymax=0.9,color=colors[i],linestyle=':',linewidth=1)
                plt.axvline(x=bracks[i][1],ymin=0,ymax=0.9,color=colors[i],linestyle=':',linewidth=1)

            plt.legend(title = 'Routes', prop={'size':10})
            plt.title(f'Total Availability - {title}')
            plt.xlabel('Available units per period')
            plt.ylabel('Frequency')
            plt.show()


    # Auxiliary function to return mean 
    @staticmethod
    def return_mean(values:list):
        if len(values)==0:
            return 0
        else:
            return (sum(values)/len(values))


    # Auxiliary function to return limits of std interval
    @staticmethod
    def return_brackets(values:list, mean:float):
        return mean + np.std(values), mean - np.std(values)


class InventoryV():

    @staticmethod
    def inventories(day:float, prod:float, inst_gen:instance_generator, real_actions:dict, actions:dict, la_decisions:dict, states:dict):

        fig, ax = plt.subplots(figsize=(12,6))

        colm = cm.get_cmap("Blues")
        col_purch = "darkorange"; col_demand = "rebeccapurple"; col_sp = "darkgreen"
        horizon = range(day+1)
        
        for t in horizon:
            if t < day: width = 0.8; delta = 0
            else: width = 0.4; delta = 0.2
            for o in range(inst_gen.O_k[prod],0,-1):
                ax.bar(x=t-delta,width=width,height=states[t][prod,o],bottom=sum(states[t][prod,oo] for oo in range(o+1,inst_gen.O_k[prod]+1)),color=colm(1-o/(inst_gen.O_k[prod]+1)))
            ax.bar(x=t-delta,width=width,height=sum(real_actions[t][1][i,prod] for i in inst_gen.Suppliers),bottom=sum(states[t][prod,o] for o in range(1,inst_gen.O_k[prod]+1)),color=col_purch)
        
        ax.bar(x=day+0.2,width=0.4,height=sum(actions[t][1][i,prod] for i in inst_gen.Suppliers),bottom=sum(states[t][prod,o] for o in range(1,inst_gen.O_k[prod]+1)),color=col_purch,alpha=0.5)
        for o in range(1,inst_gen.O_k[prod]+1):
            ax.bar(x=day+0.2,width=0.4,height=states[day][prod,o],bottom=sum(states[day][prod,oo] for oo in range(o+1,inst_gen.O_k[prod]+1)),color=colm(1-o/(inst_gen.O_k[prod]+1)),alpha=0.5)
        
        la_horizon = range(day+1,day+inst_gen.sp_window_sizes[day])
        for t in la_horizon:
            tt = t-day
            width = 0.8; delta = 0

            inv = {o:[la_decisions[day][0][tt-1][s][prod,o-1] for s in inst_gen.Samples] for o in range(1,inst_gen.O_k[prod]+1)}
            for o in range(inst_gen.O_k[prod],0,-1):
                ax.bar(x=t+delta,width=width,height=sum(inv[o][s] for s in inst_gen.Samples)/inst_gen.S,bottom=sum(sum(inv[oo][s] for s in inst_gen.Samples)/inst_gen.S for oo in range(o+1,inst_gen.O_k[prod]+1)),color=colm(1-o/(inst_gen.O_k[prod]+1)),alpha=0.5)

            z = [sum(la_decisions[day][1][tt][s][i,prod] for i in inst_gen.M_kt[prod,t]) for s in inst_gen.Samples]
            ax.bar(x=t+delta,width=width,height=sum(z[s] for s in inst_gen.Samples)/inst_gen.S,bottom=sum(sum(inv[oo][s] for s in inst_gen.Samples)/inst_gen.S for oo in range(1,inst_gen.O_k[prod]+1)),color=col_purch,alpha=0.5)

            total = [z[s]+sum(inv[o][s] for o in range(1,inst_gen.O_k[prod]+1)) for s in inst_gen.Samples]
            #ax.text(x=t+delta,y=sum(sum(inv[oo][s] for s in inst_gen.Samples)/inst_gen.S for oo in range(1,inst_gen.O_k[prod]+1)),s=f"{round(min(total),2),round(sum(total)/inst_gen.S),round(max(total),2)}")
            #ax.axvline(x=t+0.2,ymin=min(total),ymax=max(total),color="black",marker="_",mew=1.5,ms=14)
        
        # Bars for legend
        ax.bar(x=-1,width=0.1,height=1,color=col_purch,label="Purchase")
        
        for s in inst_gen.Samples:
            ax.plot([day]+list(la_horizon),[inst_gen.s_paths_d[day][t-day,s][prod] for t in [day]+list(la_horizon)],linestyle="-",color=col_sp)
            ax.plot([day-1,day],[inst_gen.W_d[day-1][prod],inst_gen.s_paths_d[day][0,s][prod]],linestyle="-",color=col_sp)
        ax.plot(horizon[:-1],[inst_gen.W_d[t][prod] for t in horizon[:-1]],marker="*",linestyle="-",markersize=16,color=col_demand,label="Historical\nDemand")
        ax.plot([-1],[-1],color=col_sp,linestyle="-",marker="",label="Sample\npaths")

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1,2,0]
        ax.legend(handles=[handles[i] for i in order],labels=[labels[i] for i in order],loc="upper left",fontsize=14)

        cmaplist = [colm(1-o/(inst_gen.O_k[prod]+1)) for o in range(inst_gen.O_k[prod],0,-1)]
        cmap = mcolors.LinearSegmentedColormap.from_list("Custom",cmaplist,inst_gen.O_k[prod])
        bounds = [o for o in range(1,inst_gen.O_k[prod]+2)]
        norm = mcolors.BoundaryNorm(bounds,inst_gen.O_k[prod])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cmm = plt.colorbar(mappable=sm,ax=ax)
        cmm.set_label(label="Product's Age",labelpad=-60,fontsize=18)
        cmm.ax.tick_params(labelsize=18)
        cmm.set_ticklabels(range(inst_gen.O_k[prod]+1,0,-1))

        ax.legend(loc="upper left",fontsize=14)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.5,inst_gen.T-0.5)
        ax.set_xlabel("Day",fontsize=20)
        ax.set_ylabel("Inventory Level (kg)",fontsize=20)
        ax.set_yticklabels([int(i) for i in ax.get_yticks()],fontsize=18)
        ticks = [i for i in range(inst_gen.T)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks,fontsize=18)

        return ax


class InstanceV():
    @staticmethod
    def plot_overlapping_distributions(q_params,d_params,num_samples=100_000,type='hist'):
        if type == 'hist':
            # Extract parameters for the uniform distribution
            q_min, q_max = q_params

            # Generate samples for the uniform distribution
            uniform_samples = np.random.uniform(q_min, q_max, num_samples)

            # Extract parameters for the log-normal distribution
            d_mean, d_dev = d_params

            # Generate samples for the log-normal distribution
            lognorm_samples = np.random.lognormal(mean=np.log(d_mean), sigma=d_dev, size=num_samples)

            # Set up figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim([0,2*q_max])

            # Plot the histograms with customized appearance
            ax.hist(uniform_samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.2, label='Availabilty (per supplier)')
            ax.hist(lognorm_samples, bins=50, density=True, alpha=0.7, color='salmon', edgecolor='black', linewidth=1.2, label='Demand')

            # Add labels and a legend
            ax.set_xlabel('Quantity', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title('Overlapping Demand and Offer Distributions', fontsize=14)
            ax.legend(fontsize=10)

            # Show a grid for better readability
            ax.grid(True, linestyle='--', alpha=0.5)

            # Show the plot
            plt.show()

        elif type == 'line':
            # Extract parameters for the uniform distribution
            q_min, q_max = q_params

            # Generate samples for the uniform distribution
            uniform_samples = np.random.uniform(q_min, q_max, num_samples)

            # Extract parameters for the log-normal distribution
            d_mean, d_dev = d_params

            # Generate samples for the log-normal distribution
            lognorm_samples = np.random.lognormal(mean=np.log(d_mean), sigma=d_dev, size=num_samples)

            # Set up figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))

            # Set x-axis range to extend until twice the maximum value of the uniform distribution
            x_max = 2 * q_max

            # Perform kernel density estimation (KDE)
            kde_uniform = KernelDensity(bandwidth=0.5).fit(uniform_samples.reshape(-1, 1))
            kde_lognorm = KernelDensity(bandwidth=0.5).fit(lognorm_samples.reshape(-1, 1))

            # Generate values for the x-axis
            x_values = np.linspace(0, x_max, 1000).reshape(-1, 1)

            # Plot KDEs with a smooth line
            ax.plot(x_values, np.exp(kde_uniform.score_samples(x_values)), label='Offer', color='skyblue')
            ax.plot(x_values, np.exp(kde_lognorm.score_samples(x_values)), label='Demand', color='salmon')

            # Add labels and a legend
            ax.set_xlabel('Quantity', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title('Overlapping Offer and Demand Distributions', fontsize=14)
            ax.legend(fontsize=10)

            # Show a grid for better readability
            ax.grid(True, linestyle='--', alpha=0.5)

            # Show the plot
            plt.show()



