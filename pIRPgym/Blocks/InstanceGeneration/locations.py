import numpy as np
from numpy.random import seed, randint
import os

class locations():

    @staticmethod
    def generate_grid(V: range): 
        # Suppliers locations in grid
        size_grid = 1000
        coor = {i:(randint(0, size_grid+1), randint(0, size_grid+1)) for i in V}
        return coor, V
    

    @staticmethod
    def euclidean_distance(coor: dict, V: range):
        # Transportation cost between nodes i and j, estimated using euclidean distance
        return {(i,j):round(np.sqrt((coor[i][0]-coor[j][0])**2 + (coor[i][1]-coor[j][1])**2)) for i in V for j in V if i!=j}
    

    @staticmethod
    def euclidean_dist_costs(V: range, d_rd_seed):
        seed(d_rd_seed + 6)
        coor, _ = locations.generate_grid(V)
        return coor, locations.euclidean_distance(coor, _)
    

    benchmarks = {'Golden_1':(5623.47,9),'Golden_2':(8404.61,10),'Golden_3':(10997.8,9),'Golden_4':(13588.6,10),
                  'Golden_5':(6460.98,5),'Golden_6':(8400.33,7), 'Golden_7':(10102.7,8),'Golden_8':(11635.3,10),
                  'Li_21':(16212.83,10),'Li_22':(14499.04,15),'Li_23':(18801.13,10),'Li_24':(21389.43,10),
                  'Li_25':(16665.7,19),'Li_26':(23977.73,10),'Li_27':(17320,20),'Li_28':(26566.03,10),
                  'Li_29':(29154.34,10),'Li_30':(31742.64,10),'Li_31':(34330.94,10),'Li_32':(37159.41,11)}

    # Uploading 
    @staticmethod
    def upload_cvrp_instance(set,instance) -> tuple:
        assert set in ['Li','Golden','Uchoa'], 'The dCVRP instance set is not available for the pIRPenv'

        if set in ['Li','Golden']: CVRPtype = 'dCVRP'; sep = ' '
        elif set == 'Uchoa': CVRPtype = 'CVRP'; sep = '\t'
        # file = open(f'../../../Instances/CVRP Instances/{CVRPtype}/{set}/{instance}', mode = 'r');     file = file.readlines()

        script_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_path, f'../../Instances/CVRP Instances/{CVRPtype}/{set}/{instance}')
        file = open(path, mode = 'r');     file = file.readlines()

        line =  int(file[3][13:17]) - 1

        M:int = int(file[3].split(' ')[-1][:-1])        # Number of suppliers
        Q:int = int(file[5].split(' ')[-1][:-1])        # Vehicles capacity

        # Max distance per route
        fila:int = 6
        d_max:float = 1e6   # Max_time
        if file[fila][0]=='D':
            d_max = float(file[fila].split(' ')[-1][:-1])
            fila += 1
        
        # Coordinates
        coor:dict = dict()
        while True:
            fila += 1
            if not file[fila][0] == 'D':
                vals = file[fila].split(sep)
                vals[2] = vals[2][:-1]
                coor[int(vals[0]) - 1] = (float(vals[1]), float(vals[2]))
            else:   break

        # Demand
        purchase:dict = dict()
        while True:
            fila += 1
            if not file[fila][0] == 'D':
                vals = file[fila].split(sep)
                if vals[0] != '1':
                    purchase[float(vals[0]) - 1] = float(vals[1])
            else:   break
        
        return M-1,Q,d_max,coor,purchase,locations.benchmarks[instance[:-4]]


    

        
