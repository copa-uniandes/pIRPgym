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
import pandas as pd


class Routing_Visualizations():
    # Sumarizes and compares routing solutions
    def compare_routing_strategies(data:dict[list]):
        print('Policy \t#Veh \tDist \tTime')
        for strategy, performance in data.items():
            if strategy != 'HyGeSe':
                print(f'{strategy} \t{len(performance[0])} \t{round(sum(performance[1]),2)} \t{round(performance[2],2)}')
            else:
                print(f'{strategy} \t{len(performance[0])} \t{round(performance[1],2)} \t{round(performance[2],2)}')
 
    # Displays the historic availability of a given route for a given product
    def route_availability_per_product(route:list, product:int, inst_gen:instance_generator, env:steroid_IRP, include_ceros:bool = False):
        series = list()
        labels = list()
        for i in route[1:-1]:
            if include_ceros:   series.append(inst_gen.hist_q[env.t][i,product])
            else:               series.append([ii for ii in inst_gen.hist_q[env.t][i,product] if ii != 0])
            
            labels.append(str(i))

        avg = Routing_Visualizations.return_mean([serie[j] for serie in series for j in range(len(serie))])
        bracket = Routing_Visualizations.return_brackets([serie[j] for serie in series for j in range(len(serie))], avg)

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