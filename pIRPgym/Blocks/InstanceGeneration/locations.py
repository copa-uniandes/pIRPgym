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
    

    # Uploading 
    @staticmethod
    def upload_cvrp_instance(set,instance) -> tuple[int,int,int,dict[float],dict[float]]:
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
        d_max:int = 1e6   # Max_time
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
        

        return M-1, Q, d_max, coor, purchase
        
