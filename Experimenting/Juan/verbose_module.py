from time import process_time
import gurobi as gu

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
 

def routing_progress():
    @staticmethod
    def print_iteration_head(strategies,show_gap=False):
        assert 'CG' not in strategies and show_gap, 'Gap can only be computed with exact solution'
        num = len(strategies)
        if show_gap: item = 'Gap'
        else:   item = 'Obj'

        print('*'*num*10 + "  pIRP environment  "+'*'*num*10,flush = True)
        print(f'{"-"*num*12} Routing {"-"*num*12}')
        string1 = '----------'
        string2 = 't t(s)\t|'
        for strategy in strategies:
            string1 += f'\t   {strategy} \t \t|'
            string1 += f't(s)\t #Veh \t {item} \t|'



        if not exclude_MIP:

            print(f'{"-"*8}| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS*\t|       MIP\t|     CG')
            print('t t(s)\t| cost \treal c.\t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t')
            print('-'*120)
        else:
            print('*'*46 + "  pIRP environment  "+'*'*46,flush=True)
            print(f'{"-"*8}|-- Purchasing -|{"-"*39} Routing {"-"*39}')
            print(f'{"-"*8}| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS*\t| \t   CG\t \t|')
            print('t t(s)\t| cost \treal c.\t| #Veh \t Obj \t| #Veh \t Obj \t| #Veh \t Obj \t| #Veh \t Obj \t| t(s)\t  #Veh \t Obj \t| ')
            print('-'*112)


class routing_instances():

    @staticmethod
    def print_head(show_gap):
        if show_gap: 
            item = 'gap'
        else:
            item = 'Obj'

        print('*'*37 + "  Routing Strategies on Classic Instances  "+'*'*37,flush = True)
        print(f'{"-"*8}|\tBKS \t|\t   NN \t \t|\t   RCL \t \t|\t   HGA \t \t|\t  HGS*')# \t \t|\t   CG \t \t')
        print(f'Inst\t| #Veh \t Obj \t| t(s)\t #Veh \t {item} \t| t(s) \t #Veh \t{item} \t| t(s) \t #Veh \t {item} \t| t(s) \t #Veh \t {item} \t|')# \t| t(s) \t #Veh \t{ item}')
        print('-'*118)

    @staticmethod
    def print_inst(set,instance,bks,k):
        if set == 'Li':
            string = f'Li {instance[-6:-4]} \t| {k} \t{round(bks)} \t|'
        else:
            string = f'Go {instance[-5:-4]} \t| {k} \t{round(bks)} \t|'
        print(string,end='\r')
        return string
    
    @staticmethod
    def print_routing_update(string,obj,veh,t,show_gap,benchmark,end=False):
        if t < 10: tt = round(t,2)
        elif t < 100: tt = round(t,1)
        else: tt = round(t)
        if not show_gap:
            if tt <100: 
                string += f' {tt} \t   {veh}\t {round(obj,1)} \t|'
            else:
                string += f' {tt}\t   {veh}\t {round(obj,1)} \t|'
        else:
            gap = round((obj-benchmark[0])/benchmark[0],4) * 100
            gap = round(gap,2)
            if tt < 100:
                string += f' {tt} \t  {veh}\t {gap} \t|'
            else:
                string += f' {tt}\t  {veh}\t {gap} \t|'
            
        
        
        if not end:
            print(string,end='\r')
        else:
            print(string)
        
        return string