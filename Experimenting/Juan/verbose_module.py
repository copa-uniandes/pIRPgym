from time import process_time
import gurobipy as gu

mmm = gu.Model()
del mmm

class complete_progress():
    @staticmethod
    def print_iteration_head(exclude_MIP=False):
        if not exclude_MIP:
            print('*'*50 + "  pIRP environment  "+'*'*50,flush = True)
            print(f'{"-"*8}|-- Purchasing -|{"-"*43} Routing {"-"*43}')
            print(f'{"-"*8}| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS*\t|       MIP\t|     CG')
            print('t t(s)\t| cost \treal c.\t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t')
            print('-'*120)
        else:
            print('*'*46 + "  pIRP environment  "+'*'*46,flush=True)
            print(f'{"-"*8}|-- Purchasing -|{"-"*39} Routing {"-"*39}')
            print(f'{"-"*8}| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS*\t| \t   CG\t \t|')
            print('t t(s)\t| cost \treal c.\t| #Veh \t Obj \t| #Veh \t Obj \t| #Veh \t Obj \t| #Veh \t Obj \t| t(s)\t  #Veh \t Obj \t| ')
            print('-'*112)

    @staticmethod
    def print_step(t,start):
        time = round(process_time()-start,2)
        if t+1 < 10:
            if time < 10:
                string = f'{t+1} {round(process_time()-start,2)}\t|'
                print(string,end='\r')
            elif time < 100:
                string = f'{t+1} {round(process_time()-start,2)} |' 
                print(string,end='\r')
            else:
                string = f'{t+1} {round(process_time()-start,1)} |'
                print(string,end='\r')
        else:
            if time < 10:
                string = f'{t+1} {round(process_time()-start,2)}\t|'
                print(string,end='\r')
            elif time < 100:
                string = f'{t+1} {round(process_time()-start,2)} |' 
                print(string,end='\r')
            else:
                string = f'{t+1} {round(process_time()-start)} |'
                print(string,end='\r')
        
        return string

    @staticmethod
    def print_purchase_update(string,prices,purchase):
        cost = sum(purchase[i,k]*prices[i,k] for (i,k) in purchase.keys())
        string += f'{round(cost)}\t{round(cost)}\t|'
        print(string,end='\r')
        return string

    @staticmethod
    def print_routing_update(string,FO,veh,end=False,CG_time=None):
        if CG_time != None:
            if CG_time < 1000:
                string += f' {round(CG_time,1)}\t{round(FO)} \t  {veh}\t'
            else:
                string += f' {round(CG_time)}\t{round(FO)} \t  {veh}\t'
        elif FO < 10000:
            string += f' {round(FO)} \t  {veh}\t'
        else:
            string += f' {round(FO)}\t  {veh}\t'
        string += '|'

        if not end:
            print(string,end='\r')
        else:
            print(string)
        return string
 

class routing_progress():

    @staticmethod
    def print_iteration_head(policies:list,show_gap=False):
        assert not ('CG' not in policies and show_gap), 'Gap can only be computed with exact solution'
        num = len(policies)
        if show_gap: item = 'gap'
        else:   item = 'Obj'

        print(f'***{"*"*num*11}  pIRP environment  {"*"*num*11}**',flush = True)
        print(f'---{"-"*num*12} Routing {"-"*num*12}-----')
        string1 = '--------|-------|'
        string2 = 't t(s)\t|   N\t|'
        for strategy in policies:
            if strategy == 'CG':
                string1 += f'\t  {strategy} \t \t|'
                string2 += f' t(s)\t #Veh \t obj \t|'
            else:
                string1 += f'\t  {strategy} \t \t|'
                string2 += f' t(s)\t #Veh \t {item} \t|'

        print(string1)
        print(string2)
        print(f'-----------------{"-"*num*24}')
    
    @staticmethod
    def print_step(t,start,purchase):
        num_suppliers = len(set(key[0] for key in purchase.keys() if purchase[key]>0))
        time = round(process_time()-start,2)
        if t+1 < 10:
            if time < 10:
                string = f'{t+1} {round(process_time()-start,2)}\t|   {num_suppliers}\t|'
                print(string,end='\r')
            elif time < 100:
                string = f'{t+1} {round(process_time()-start,2):.2f} |   {num_suppliers}\t|' 
                print(string,end='\r')
            elif time < 1000:
                string = f'{t+1} {round(process_time()-start,1):.1f} |   {num_suppliers}\t|'
                print(string,end='\r')
            else:
                string = f'{t+1} {round(process_time()-start):.0f} |   {num_suppliers}\t|'
                print(string,end='\r')
        else:
            if time < 10:
                string = f'{t+1} {round(process_time()-start,2)}\t|   {num_suppliers}\t|'
                print(string,end='\r')
            elif time < 100:
                string = f'{t+1} {round(process_time()-start,2):.2f} |   {num_suppliers}\t|' 
                print(string,end='\r')
            elif time < 1000:
                string = f'{t+1} {round(process_time()-start):.1f} |   {num_suppliers}\t|'
                print(string,end='\r')
            else:
                string = f'{t+1} {round(process_time()-start):.0f} |   {num_suppliers}\t|'
                print(string,end='\r')
        
        return string

    @staticmethod
    def print_routing_update(string,time,vehicles,objective,end=False,CG_obj=None):
        if CG_obj==None:
            if time < 1000:
                string += f' {round(time,1)}\t  {round(vehicles,1)} \t {round(objective)}\t|'
            else:
                string += f' {round(time)}\t  {round(vehicles,1)} \t {round(objective)}\t|'
        else:
            gap = round((objective-CG_obj)/CG_obj,4)
            if time < 1000:
                string += f' {round(time,1)}\t  {round(vehicles,1)} \t {round(gap*100,2)}\t|'
            else:
                string += f' {round(time)}\t  {round(vehicles,1)} \t {round(gap*100,2)}\t|'

        if not end:
            print(string,end='\r')
        else:
            print(string)
        return string


class routing_instances():

    @staticmethod
    def print_head(policies:list,inst_set:str,show_gap:bool):
        if show_gap: item = 'gap'
        else: item = 'Obj' 
        num = len(policies)

        print(f'*****************{"*"*num*13}  {inst_set} set Instances  {"*"*num*13}*****************',flush = True)
        string1 = f'--------|{"-"*23}|'
        string2 = 'Inst\t|   M \t  Veh\t Obj\t|'
        for strategy in policies:
            if strategy not in  ['RCL']:
                string1 += f'\t  {strategy} \t \t|'
                string2 += f' t(s)\t #Veh \t {item} \t|'
            else:
                string1 += f'\t  \t  \t \t{strategy} \t \t \t|'
                string2 += f' t(s)\t #Veh \t mean \tmedian\t stdev\t min\t max\t|'
                
        print(string1)
        print(string2)
        print(f'--------------------------------------------------------{"-"*num*28}')


        # print(f'{"-"*8}|\tBKS \t|\t   NN \t \t|\t   RCL \t \t|\t   HGA \t \t|\t  HGS*')# \t \t|\t   CG \t \t')
        # print(f'Inst\t| #Veh \t Obj \t| t(s)\t #Veh \t {item} \t| t(s) \t #Veh \t{item} \t| t(s) \t #Veh \t {item} \t| t(s) \t #Veh \t {item} \t|')# \t| t(s) \t #Veh \t{ item}')
        # print('-'*118)

    @staticmethod
    def print_inst(set,instance,M,k,bks):
        if set == 'Li':
            string = f'Li {instance[-6:-4]} \t|  {M}\t  {k} \t{round(bks)} \t|'
        elif set == 'Golden':
            string = f'Go {instance[-5:-4]} \t|  {M}\t  {k} \t{round(bks)} \t|'
        elif set == 'Uchoa':
            string = f'{instance[2:-4]}|  {M}\t  {k} \t{round(bks)} \t|' 
        else:
            string = f'{instance}|  {M}\t  {k} \t{round(bks)} \t|' 
        print(string,end='\r')
        return string
    
    @staticmethod
    def print_routing_update(string,obj,veh,t,show_gap,benchmark,intervals=False,end=False):
        if t < 10: tt = round(t,2)
        elif t < 100: tt = round(t,1)
        else: tt = round(t)

        if type(veh) != int:
            veh = round(veh,1)
        if not show_gap:
            if tt <10: 
                string += f' {tt:.2f}\t   {veh}\t {round(obj,1)}\t|'
            else:
                string += f' {tt:.1f}\t   {veh}\t {round(obj,1)}\t|'
        else:
            gap = round((obj-benchmark[0])/benchmark[0],4) * 100; gap = round(gap,2)

            if intervals == False:
                if tt < 10:
                    string += f' {tt:.2f}\t  {veh}\t {gap}\t|'
                else:
                    string += f' {tt:.1f}\t  {veh}\t {gap}\t|'
            else:
                median = round((intervals[0]-benchmark[0])/benchmark[0],4) * 100; median = round(median,2)
                stdev = round(intervals[1]/benchmark[0],4) * 100; stdev = round(stdev,2)
                min_gap = round((intervals[2]-benchmark[0])/benchmark[0],4) * 100; min_gap = round(min_gap,2)
                max_gap = round((intervals[3]-benchmark[0])/benchmark[0],4) * 100; max_gap = round(max_gap,2)
                if tt < 10:
                    string += f' {tt:.2f}\t  {veh}\t {gap}\t {median} \t{stdev}\t  {min_gap}\t {max_gap}\t|'
                else:
                    string += f' {tt:.1f}\t  {veh}\t {gap}\t {median} \t{stdev}\t  {min_gap}\t {max_gap}\t|'
            
        
        
        if not end:
            print(string,end='\r')
        else:
            print(string)
        
        return string
    
    @staticmethod
    def print_comparison_head(policies:list,inst_set:str,show_gap:bool):
        if show_gap: item = 'gap'
        else: item = 'Obj' 
        num = len(policies)

        print(f'**{"*"*num*13}  {inst_set} set Instances  {"*"*num*13}***',flush = True)
        string1 = f'----------------|'
        string2 = 'Sizes\t  Num \t|'
        for strategy in policies:
            if strategy not in ['RCL']:
                string1 += f'\t{strategy} \t|'
                string2 += f' t(s) \t   {item} \t|'
            else:
                string1 += f'\t     {strategy} \t \t|'
                string2 += f' t(s)\t  mean\t  min\t   max\t|'
                
        print(string1)
        print(string2)
        print(f'---------{"-"*num*28}')


    @staticmethod
    def print_comparison_inst(sizes,num_instances):

        if num_instances>1:
            string = f'{sizes[0]}-{sizes[1]}\t    {num_instances}\t|' 
        else:
            string = f'> {sizes[0]}\t    {num_instances}\t|' 
        print(string,end='\r')
        return string
    
    @staticmethod
    def print_routing_comparison_update(string,nn_gap,nn_time,RCL_gap,RCL_time,RCL_min,RCL_max):
        string += f' {nn_time:.2f}\t  {round(nn_gap*100,2)}\t| {RCL_time:.2f}\t  {round(RCL_gap*100,2)}\t  {round(RCL_min*100,2)}\t {round(RCL_max*100,2)}\t|'        
        print(string)
        
        return string


class CG_initialization():
    @staticmethod
    def print_head(experiment,replica):
        print('*'*89 + f"  Heuristic Init on CG - Exp{experiment}/Replica{replica} "+'*'*88,flush = True)
        print(f'{"-"*8}|\tNN\t|\t  \tCG \t \t|\t  CG w/ Init (alpha = 0.1) \t|\t  CG w/ Init (alpha = 0.2) \t|\t  CG w/ Init (alpha = 0.4) \t|\t  CG w/ Init (alpha = 0.6) \t|')
        print(f't   M\t| #Veh\t  Obj\t| t(s)\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t|')
        print('-'*216)

    @staticmethod
    def print_step(t,purchase,nn_veh,nn_obj,):
        num_suppliers = len(set(key[0] for key in purchase.keys() if purchase[key]>0))
        if t < 10:
            string = f'{t}   {num_suppliers}\t| {nn_veh}\t {nn_obj}\t|'
        else:
            string = f'{t}  {num_suppliers}\t| {nn_veh}\t {nn_obj}\t|'
        print(string,end='\r')
        return string
    
    @staticmethod
    def print_update(string,t,cols,veh,obj,end=False):
        if type(cols) == int:
            if t < 100:
                string += f' {t:.2f}\t  {cols}\t {veh}\t{round(obj,1)}\t|'
            elif t < 1000:
                string += f' {t:.1f}\t  {cols}\t {veh}\t{round(obj,1)}\t|'
            else:
                string += f' {t:.0f}\t  {cols}\t {veh}\t{round(obj,1)}\t|'
            if end:
                print(string)
                return string
            else:
                print(string,end='\r')
                return string
        else:
            if t < 100:
                string += f' {t:.2f}\t  {cols[0]}\t  {cols[1]}\t {veh}\t{round(obj,1)}\t|'
            elif t < 1000:
                string += f' {t:.1f}\t  {cols[0]}\t  {cols[1]}\t {veh}\t{round(obj,1)}\t|'
            else:
                string += f' {t:.0f}\t  {cols[0]}\t  {cols[1]}\t {veh}\t{round(obj,1)}\t|'
            if end:
                print(string)
                return string
            else:
                print(string,end='\r')
                return string
        


















********************************************************  Li set Instances  ********************************************************
--------|-----------------------|	  NN 	 	|	  	  	 	RCL 	 	 	|	  GA 	 	|
Inst	|   M 	  Veh	 Obj	| t(s)	 #Veh 	 gap 	| t(s)	 #Veh 	 mean 	median	 stdev	 min	 max	| t(s)	 #Veh 	 gap 	|
--------------------------------------------------------------------------------------------------------------------------------------------
Li 21 	|  560	  10 	16213 	| 0.03	  13	 28.0	| 0.16	  14.8	 46.63	 35.18 	27.38	  24.5	 130.31	| 3.05 & 13 & 28.0	|
Li 22 	|  600	  15 	14499 	| 0.00	  22	 31.72	| 0.09	  24.3	 47.54	 42.62 	11.58	  36.62	 74.5	| 3.20 & 22 & 31.72	|
Li 23 	|  640	  10 	18801 	| 0.00	  14	 41.85	| 0.13	  13.8	 43.66	 32.81 	30.65	  20.93	 151.15	| 3.08 & 14 & 41.85	|
Li 24 	|  720	  10 	21389 	| 0.06	  13	 30.59	| 0.16	  14.8	 49.19	 35.24 	34.23	  26.75	 168.69	| 3.30 & 13 & 30.59	|
Li 25 	|  760	  19 	16666 	| 0.05	  28	 31.56	| 0.16	  30.1	 40.96	 39.63 	7.12	  31.74	 68.16	| 3.14 & 28 & 31.56	|
Li 26 	|  800	  10 	23978 	| 0.03	  14	 27.61	| 0.20	  16.1	 55.42	 36.09 	39.46	  25.56	 183.53	| 3.34 & 14 & 27.61	|
Li 27 	|  840	  20 	17320 	| 0.05	  29	 26.82	| 0.22	  32.1	 44.38	 37.0 	18.0	  32.04	 114.97	| 3.23 & 29 & 26.82	|
Li 28 	|  880	  10 	26566 	| 0.08	  14	 29.18	| 0.26	  15.0	 42.83	 32.27 	37.65	  20.99	 203.73	| 3.35 & 14 & 29.18	|
Li 29 	|  960	  10 	29154 	| 0.11	  14	 32.12	| 0.31	  15.6	 49.14	 33.56 	43.6	  22.12	 219.43	| 3.19 & 14 & 32.12	|
Li 30 	|  1040	  10 	31743 	| 0.06	  14	 30.07	| 0.58	  15.8	 46.86	 34.71 	34.98	  26.8	 140.76	| 3.28 & 14 & 30.07	|
Li 31 	|  1120	  10 	34331 	| 0.17	  13	 24.0	| 0.84	  17.4	 66.06	 37.74 	66.12	  24.8	 242.71	| 3.50 & 13 & 24.0	|
Li 32 	|  1200	  11 	37159 	| 0.16	  12	 9.44	| 1.30	  17.4	 58.52	 36.39 	53.03	  25.6	 244.79	| 3.95 & 12 & 9.44	|


********************************************************  Golden set Instances  ********************************************************
--------|-----------------------|	  NN 	 	|	  	  	 	RCL 	 	 	|	  GA 	 	|
Inst	|   M 	  Veh	 Obj	| t(s)	 #Veh 	 gap 	| t(s)	 #Veh 	 mean 	median	 stdev	 min	 max	| t(s)	 #Veh 	 gap 	|
--------------------------------------------------------------------------------------------------------------------------------------------
Go 1 	|  240	  9 	5623 	| 0.00	  14	 32.18	| 0.03	  13.9	 35.84	 33.82 	8.86	  27.02	 68.95	| 3.01 & 14 & 32.18	|
Go 2 	|  320	  10 	8405 	| 0.02	  15	 33.93	| 0.03	  15.4	 41.52	 36.38 	14.85	  28.49	 87.19	| 3.06 & 15 & 33.93	|
Go 3 	|  400	  9 	10998 	| 0.00	  15	 39.93	| 0.05	  15.1	 47.81	 36.8 	19.74	  30.95	 102.04	| 3.05 & 15 & 39.93	|
Go 4 	|  480	  10 	13589 	| 0.02	  13	 31.83	| 0.07	  13.3	 33.77	 30.16 	11.18	  23.46	 76.41	| 3.08 & 13 & 31.83	|
Go 5 	|  200	  5 	6461 	| 0.00	  6	    49.44	| 0.02	  6.0	 38.1	 34.54 	12.36	  19.36	 89.29	| 15.7 & 5 & 29.56	|
Go 6 	|  280	  7 	8400 	| 0.00	  8	    22.97	| 0.02	  8.8	 36.53	 30.53 	20.45	  19.46	 102.42	| 3.03 & 8 & 22.97	|
Go 7 	|  360	  8 	10103 	| 0.00	  11	 23.5	| 0.03	  12.3	 36.55	 34.12 	9.18	  24.52	 70.28	| 3.06 & 11 & 23.5	|
Go 8 	|  440	  10 	11635 	| 0.00	  15	 34.25	| 0.05	  16.5	 51.46	 40.47 	22.54	  30.52	 103.99	| 3.10 & 15 & 34.25	|



mean = [28.0, 31.72, 41.85, 30.59, 31.56, 27.61, 26.82, 29.18, 32.12, 30.07, 24.0, 9.44]

mean2 = [32.18, 33.93, 39.93, 31.83, 49.44, 22.97, 23.5, 34.25]


median = [35.18, 42.62, 32.81, 35.24, 39.63, 36.09, 37.0, 32.27, 33.56, 34.71, 37.74, 36.39]
median2 = [33.82,36.38,36.8,30.16,34.54,30.53,34.12,40.47]

times2 = [0.03,0.03,0.05,0.07,0.02,0.02,0.03,0.05]


    tt = [0.00,0.02,0.00,0.02,0.00,0.00,0.00,0.00]