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

        print(f'**************{"*"*num*13}  {inst_set} set Instances  {"*"*num*13}**************',flush = True)
        string1 = f'--------|{"-"*23}|'
        string2 = 'Inst\t|   M \t  Veh\t Obj\t|'
        for strategy in policies:
            if strategy not in  ['RCL']:
                string1 += f'\t  {strategy} \t \t|'
                string2 += f' t(s)\t #Veh \t {item} \t|'
            else:
                string1 += f'\t  \t  \t{strategy} \t \t \t|'
                string2 += f' t(s)\t #Veh \t {item} \t stdev\t min\t max\t|'
                
        print(string1)
        print(string2)
        print(f'------------------------------------------------{"-"*num*28}')


        # print(f'{"-"*8}|\tBKS \t|\t   NN \t \t|\t   RCL \t \t|\t   HGA \t \t|\t  HGS*')# \t \t|\t   CG \t \t')
        # print(f'Inst\t| #Veh \t Obj \t| t(s)\t #Veh \t {item} \t| t(s) \t #Veh \t{item} \t| t(s) \t #Veh \t {item} \t| t(s) \t #Veh \t {item} \t|')# \t| t(s) \t #Veh \t{ item}')
        # print('-'*118)

    @staticmethod
    def print_inst(set,instance,M,k,bks):
        if set == 'Li':
            string = f'Li {instance[-6:-4]} \t|  {M}\t  {k} \t{round(bks)} \t|'
        elif set == 'Golden':
            string = f'Go {instance[-5:-4]} \t|  {M}\t  {k} \t{round(bks)} \t|'
        else:
            string = f'{instance[2:-4]}|  {M}\t  {k} \t{round(bks)} \t|' 
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
                stdev = round(intervals[0]/benchmark[0],4) * 100; stdev = round(stdev,2)
                min_gap = round((intervals[1]-benchmark[0])/benchmark[0],4) * 100; min_gap = round(min_gap,2)
                max_gap = round((intervals[2]-benchmark[0])/benchmark[0],4) * 100; max_gap = round(max_gap,2)
                if tt < 10:
                    string += f' {tt:.2f}\t  {veh}\t {gap}\t {stdev}\t  {min_gap}\t {max_gap}\t|'
                else:
                    string += f' {tt:.1f}\t  {veh}\t {gap}\t {stdev}\t  {min_gap}\t {max_gap}\t|'
            
        
        
        if not end:
            print(string,end='\r')
        else:
            print(string)
        
        return string
    

class CG_initialization():
    @staticmethod
    def print_head(experiment,replica):
        print('*'*89 + f"  Heuristic Init on CG - Exp{experiment}/Replica{replica} "+'*'*88,flush = True)
        print(f'{"-"*8}|\tNN\t|\t  \tCG \t \t|\t  CG w/ Init (alpha = 0.1) \t|\t  CG w/ Init (alpha = 0.2) \t|\t  CG w/ Init (alpha = 0.4) \t|\t  CG w/ Init (alpha = 0.6) \t|')
        print(f't    M\t| #Veh\t  Obj\t| t(s)\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t| t(s)\tRCLcols\t cols\t #Veh \t Obj \t|')
        print('-'*216)

    @staticmethod
    def print_step(t,purchase,nn_veh,nn_obj,):
        num_suppliers = len(set(key[0] for key in purchase.keys() if purchase[key]>0))
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
        
