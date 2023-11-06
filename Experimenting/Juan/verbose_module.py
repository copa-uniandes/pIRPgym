from time import process_time
import gurobi as gu

mmm = gu.Model()
del mmm

def print_iteration_head():
    print('*'*50 + "  pIRP environment  "+'*'*50,flush = True)
    print(f'{"-"*8}|-- Purchasing -|{"-"*43} Routing {"-"*43}')
    print(f'{"-"*8}| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS*\t|       MIP\t|     CG')
    print('t t(s)\t| cost \treal c.\t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t')
    print('-'*120)


def print_step(t,start):
    time = round(process_time()-start,2)
    if t < 10:
        if time < 10:
            string = f'{t+1} {round(process_time()-start,2)}\t|'
            print(string,end='\r')
        elif time < 100:
            string = f'{t+1} {round(process_time()-start,2)} |' 
            print(string,end='\r')
        else:
            string = f'{t+1} {round(process_time()-start,2)}|'
            print(string,end='\r')
    else:   
        string = f'{t+1} {round(process_time()-start,2)}\t|'
        print(string,end='\r')
    
    return string


def print_purchase_update(string,prices,purchase):
    cost = sum(purchase[i,k]*prices[i,k] for (i,k) in purchase.keys())
    string += f'{round(cost)}\t{round(cost)}\t|'
    print(string,end='\r')
    return string


def print_routing_update(string,FO,veh,end=False):
    string += f' {round(FO)} \t  {veh}\t'
    if not end:
        string += '|'
        print(string,end='\r')
    else:
        print(string)
    return string
 


class routing_instances():

    @staticmethod
    def print_head():
        print('*'*50 + "  Routing Strategies  "+'*'*50,flush = True)
        print(f'{"-"*8}|-- Purchasing -|{"-"*43} Routing {"-"*43}')
        print(f'{"-"*8}| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS*\t|       MIP\t|     CG')
        print('t t(s)\t| cost \treal c.\t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t')
        print('-'*120)