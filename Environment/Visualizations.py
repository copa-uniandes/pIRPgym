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
from matplotlib import cm
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


class PerishabilityV():
    pass