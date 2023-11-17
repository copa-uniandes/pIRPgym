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
                  'Li_29':(29154.34,10),'Li_30':(31742.64,10),'Li_31':(34330.94,10),'Li_32':(37159.41,11),
                  
                  'X-n101-k25':(27591,25),'X-n106-k14':(26362,14),'X-n110-k13':(14971,13),
                  'X-n115-k10':(12747,10),'X-n120-k6':(13332,6),'X-n125-k30':(55539,30),
                  'X-n129-k18':(28940,18),'X-n134-k13':(10916,13),'X-n139-k10':(13590,10),
                  'X-n143-k7':(15700,7),'X-n148-k46':(43448,46),'X-n153-k22':(21220,22),
                  'X-n157-k13':(16876,13),'X-n162-k11':(14138,11),'X-n167-k10':(20557,10),
                  'X-n172-k51':(45607,51),'X-n176-k26':(47812,26),'X-n181-k23':(25569,23),
                  'X-n186-k15':(24145,15),'X-n190-k8':(16980,8),'X-n195-k51':(44225,51),
                  'X-n200-k36':(58578,36),'X-n204-k19':(19565,19),'X-n209-k16':(30656,16),
                  'X-n214-k11':(10856,11),'X-n219-k73':(117595,73),'X-n223-k34':(40437,34),
                  'X-n228-k23':(25742,23),'X-n233-k16':(19230,16),'X-n237-k14':(27042,14),
                  'X-n242-k48':(82751,48),'X-n247-k50':(37274,47),'X-n251-k28':(38684,28),
                  'X-n256-k16':(18839,16),'X-n261-k13':(26558,13),'X-n266-k58':(75478,58),
                  'X-n270-k35':(35291,35),'X-n275-k28':(21245,28),'X-n280-k17':(33503,17),
                  'X-n284-k15':(20215,15),'X-n289-k60':(95151,60),'X-n294-k50':(47161,50),
                  'X-n298-k31':(34231,31),'X-n303-k21':(21736,21),'X-n308-k13':(25859,13),
                  'X-n313-k71':(94043,71),'X-n317-k53':(78355,53),'X-n322-k28':(29834,28),
                  'X-n327-k20':(27532,20),'X-n331-k15':(31102,15),'X-n336-k84':(139111,84),
                  'X-n344-k43':(42050,43),'X-n351-k40':(25896,40),'X-n359-k29':(51505,29),
                  'X-n367-k17':(22814,17),'X-n376-k94':(147713,94),'X-n384-k52':(65928,52),
                  'X-n393-k38':(38260,38),'X-n401-k29':(66154,29),'X-n411-k19':(19712,19),
                  'X-n420-k130':(107798,130),'X-n429-k61':(65449,61),'X-n439-k37':(36391,37),
                  'X-n449-k29':(55233,29),'X-n459-k26':(24139,26),'X-n469-k138':(221824,138),
                  'X-n480-k70':(89449,70),'X-n491-k59':(66483,59),'X-n502-k39':(69226,39),
                  'X-n513-k21':(24201,21),'X-n524-k153':(154593,153),'X-n536-k96':(94846,96),
                  'X-n548-k50':(86700,50),'X-n561-k42':(42717,42),'X-n573-k30':(50673,30),
                  'X-n586-k159':(190316,159),'X-n599-k92':(108451,92),'X-n613-k62':(59535,62),
                  'X-n627-k43':(62164,43),'X-n641-k35':(63682,35),'X-n655-k131':(106780,131),
                  'X-n670-k130':(146332,130),'X-n685-k75':(68205,75),'X-n701-k44':(81923,44),
                  'X-n716-k35':(43373,35),'X-n733-k159':(136187,159),'X-n749-k98':(77269,98),
                  'X-n766-k71':(114417,71),'X-n783-k48':(72386,48),'X-n801-k40':(73305,40),
                  'X-n819-k171':(158121,171),'X-n837-k142':(193737,142),'X-n856-k95':(88965,95),
                  'X-n876-k59':(99299,59),'X-n895-k37':(53860,37),'X-n916-k207':(329179,207),
                  'X-n936-k151':(132715,151),'X-n957-k87':(85465,87),'X-n979-k58':(118976,58),
                  'X-n1001-k43':(72355,43)}

    # Uploading 
    @staticmethod
    def upload_cvrp_instance(set,instance) -> tuple:
        assert set in ['Li','Golden','Uchoa'], 'The dCVRP instance set is not available for the pIRPenv'

        if set in ['Li','Golden']: CVRPtype = 'dCVRP'; sep = ' '
        elif set == 'Uchoa': CVRPtype = 'CVRP'; sep = '\t'
        # file = open(f'../../../Instances/CVRP Instances/{CVRPtype}/{set}/{instance}', mode = 'r');     file = file.readlines()

        script_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_path, f'../../Instances/CVRP Instances/{CVRPtype}/{set}/{instance}')
        file = open(path,mode='r');     file = file.readlines()

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


    

        
