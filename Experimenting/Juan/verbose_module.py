def print_iteration_head():
    print("--------- pIRP environment ---------\n",flush = True)
    print('\t|  Purchasing   | \t \t \t \t \t \tRouting')
    print('\t| Stochastic RH |       NN \t|       RCL \t|       HGA \t|       HGS \t|       MIP\t|     CG')
    print('t  t(s)\t| cost \treal c.\t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t| Obj \t #Veh \t')
    print('-'*120)