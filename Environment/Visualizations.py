"""
@author: Juan Betancourt
"""
################################## Modules ##################################
### SC classes
from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP

### Basic Librarires
import matplotlib.pyplot as plt


class Routing_Visualizations():

    # Displays the historic availability of a given route
    def route_availability(route:list, inst_gen:instance_generator, env:steroid_IRP):
        series = list()
        for i in route[1:-1]:
            series.append(inst_gen.hist_q[env.t][i,0])
        
        plt.hist(series, density = True, histtype = 'bar', label = route[1:-1])
        plt.title('Availability')
        plt.xlabel('Available units per period')
        plt.ylabel('Frequency')
        plt.show()