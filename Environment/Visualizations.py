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


class Routing_Visualizations():

    # Displays the historic availability of a given route for a given product
    def route_availability_per_product(route:list, product:int, inst_gen:instance_generator, env:steroid_IRP):
        series = list()
        labels = list()
        for i in route[1:-1]:
            series.append(inst_gen.hist_q[env.t][i,product])
            labels.append(str(i))

        avg = Routing_Visualizations.return_mean([serie[j] for serie in series for j in range(len(serie))])
        bracket = Routing_Visualizations.return_brackets([serie[j] for serie in series for j in range(len(serie))], avg)

        plt.hist(series, density = True, histtype = 'bar', label = labels)
        
        plt.axvline(x = avg, ymin = 0, ymax = 0.9, color = 'red', label = 'Mean')

        plt.legend(title = 'Suppliers', prop={'size':10})
        plt.title(f'Availability of product {product}')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()
    

    # Displays the historic total availability of the suppiers of a given route
    def route_total_availability(route:list, inst_gen:instance_generator, env:steroid_IRP):
        series = list()
        labels = list()
        for i in route[1:-1]:
            vals = list()
            for k in inst_gen.Products:
                vals.extend(inst_gen.hist_q[env.t][i,k])    
            series.append(vals)
            labels.append(str(i))
        
        avg = Routing_Visualizations.return_mean([serie[j] for serie in series for j in range(len(serie))])
        
        plt.hist(series, density = True, histtype = 'bar', label = labels)

        plt.axvline(x = avg, ymin = 0, ymax = 0.9, color = 'red', label = 'Mean')

        plt.legend(title = 'Suppliers', prop={'size':10})
        plt.title(f'Total Availability')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()
    

    # Displays the historic avaiability of different routes for a given product
    def routes_availability_per_product(routes:list, product:int, inst_gen:instance_generator, env:steroid_IRP):
        series = list()
        labels = list()
        avgs = list()
        cols = mcolors.TABLEAU_COLORS
        # colors = [cols[key] for i, key in enumerate(list(cols.keys())) if i < len(routes)]
        colors = list()

        for i, route in enumerate(routes):
            series.append([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]])
            labels.append(str(i))
            avgs.append(Routing_Visualizations.return_mean([j for i in route[1:-1] for j in inst_gen.hist_q[env.t][i,product]]))
            colors.append(list(cols.values())[i])

        # bracket = Routing_Visualizations.return_brackets([serie[j] for serie in series for j in range(len(series))], avg)
        
        plt.hist(series, density = True, histtype = 'bar', color = colors, label = labels)
        
        for i, route in enumerate(routes):
            plt.axvline(x = avgs[i], ymin = 0, ymax = 0.9, color = colors[i])

        plt.legend(title = 'Routes', prop={'size':10})
        plt.title(f'Availability of product {product}')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()
    

    # Displays the historic total avaiability of different routes
    def routes_total_availability(routes:list, inst_gen:instance_generator, env:steroid_IRP):
        series = list()
        labels = list()
        avgs = list()
        cols = mcolors.TABLEAU_COLORS
        # colors = [cols[key] for i, key in enumerate(list(cols.keys())) if i < len(routes)]
        colors = list()

        for i, route in enumerate(routes):
            vals = list()
            for j in route[1:-1]:
                for k in inst_gen.Products:
                    vals.extend(inst_gen.hist_q[env.t][j,k]) 

            series.append(vals) 
            labels.append(str(i))
            avgs.append(Routing_Visualizations.return_mean(vals))
            colors.append(list(cols.values())[i])



        plt.hist(series, density = True, histtype = 'bar', label = labels)

        for i, route in enumerate(routes):
            plt.axvline(x = avgs[i], ymin = 0, ymax = 0.9, color = colors[i])

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