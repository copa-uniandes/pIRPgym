from time import process_time
import gurobipy as gu
import numpy as np
from pickle import dump, load
import os

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
    def print_routing_comparison_update(string,nn_gap,nn_time,RCL_gap,RCL_time,RCL_min,RCL_max,GA_time,GA_gap):
        string += f' {nn_time:.2f}\t  {round(nn_gap*100,2)}\t| {RCL_time:.2f}\t  {round(RCL_gap*100,2)}\t  {round(RCL_min*100,2)}\t {round(RCL_max*100,2)}\t| {GA_time:.2f}\t {round(GA_gap*100,2)}\t|'        
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
        

class headers():

    @staticmethod
    def print_simple_header(w):
        print("\n---------------------------------------------------------------------------------------------------------------------------------------")
        print(" "*int(np.floor((135-len(w)-11)/2)) + f"Optimizing {w}" + " "*int(np.ceil((135-len(w)-11)/2)))
        print("---------------------------------------------------------------------------------------------------------------------------------------")
    
    @staticmethod
    def print_header(s):
        print("\n------------------------------------------------------")
        print(" "*int(np.floor((54-len(s))/2)) + s + " "*int(np.ceil((54-len(s))/2))  )
        print("------------------------------------------------------\n")


class objectives_performance():

    @staticmethod
    def show_normalization(inst_gen,env):

        print("\t"+" "*8,*[" "*int(np.floor((15-len(e))/2))+e+" "*int(np.ceil((15-len(e))/2))+"\t" for e in ["costs"]+inst_gen.E],sep="\t")
        for e in ["costs"]+inst_gen.E:
            print("\t"+e+"*"+" "*(8-len(e)-1),*[f"{env.payoff_matrix[e][ee]:.2e}" + f" ({(env.payoff_matrix[e][ee]-env.norm_matrix[ee]['best'])/(env.norm_matrix[ee]['worst']-env.norm_matrix[ee]['best']):.2f}) " for ee in ["costs"]+inst_gen.E],sep="\t")
    
    @staticmethod
    def show_balanced_solution(inst_gen,env,la_decisions):
        
        inventory, purchase, backorders, flow_x, flow_y = la_decisions[0], la_decisions[1], la_decisions[2], la_decisions[4], la_decisions[5]; impact = dict()

        purch_cost = sum(inst_gen.W_p[env.t][i,k]*purchase[t][s][i,k] for t in purchase for s in inst_gen.Samples for k in inst_gen.Products for i in inst_gen.M_kt[k,env.t + t])/len(inst_gen.Samples)
        backo_cost = sum(inst_gen.back_o_cost[k]*backorders[t][s][k] for t in backorders for s in inst_gen.Samples for k in inst_gen.Products)/len(inst_gen.Samples)
        rout_cost = sum(inst_gen.c[i,j]*flow_x[t][s][i,j] for (i,j) in inst_gen.A for t in flow_x for s in inst_gen.Samples)/len(inst_gen.Samples)
        holding_cost = sum(inst_gen.W_h[t][k]*inventory[t][s][k,o] for t in inventory for s in inst_gen.Samples for k in inst_gen.Products for o in range(inst_gen.O_k[k]))/len(inst_gen.Samples)
        if inst_gen.hold_cost: impact["costs"] = purch_cost + backo_cost + rout_cost + holding_cost
        else: impact["costs"] = purch_cost + backo_cost + rout_cost
        
        for e in inst_gen.E:
            transport = sum(inst_gen.c_LCA[e][k][i,j]*flow_y[t][s][i,j,k] for t in purchase for s in inst_gen.Samples for k in inst_gen.Products for (i,j) in inst_gen.A)/len(inst_gen.Samples)
            storage = sum(inst_gen.h_LCA[e][k]*inventory[t][s][k,o] for t in inventory for s in inst_gen.Samples for k in inst_gen.Products for o in range(inst_gen.O_k[k]))/len(inst_gen.Samples)
            waste = sum(inst_gen.waste_LCA[e][k]*inventory[t][s][k,inst_gen.O_k[k]] for k in inst_gen.Products for t in inventory for s in inst_gen.Samples)/len(inst_gen.Samples)

            impact[e] = transport + storage + waste

        print("\n\t"+" "*8,*[" "*int(np.floor((15-len(e))/2))+e+" "*int(np.ceil((15-len(e))/2))+"\t" for e in ["costs"]+inst_gen.E],sep="\t")
        print("\tresults ",*[f"{impact[e]:.2e}" + f" ({(impact[e]-env.norm_matrix[e]['best'])/(env.norm_matrix[e]['worst']-env.norm_matrix[e]['best']):.2f}) " for e in ["costs"]+inst_gen.E],sep="\t")


class export_results():

    @staticmethod
    def export_rewards(weights,seed_ix,rewards, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Rewards/Rewards_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Rewards_{weights}_{seed_ix}","wb")
        dump(rewards,file); file.close()
    
    @staticmethod
    def export_actions(weights,seed_ix,action, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Actions/Actions_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Actions_{weights}_{seed_ix}","wb")
        dump(action,file); file.close()
    
    @staticmethod
    def export_lookahead_decisions(weights,seed_ix,lookahead, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Lookahead/Lookahead_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Lookahead_{weights}_{seed_ix}","wb")
        dump(lookahead,file); file.close()
    
    @staticmethod
    def export_instance_parameters(weights,seed_ix,inst_gen, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Instance/Instance_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Instance_{weights}_{seed_ix}","wb")
        dump(inst_gen,file); file.close()

    @staticmethod
    def export_inventory(weights, seed_ix, i0, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Inventory/Inventory_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Inventory_{weights}_{seed_ix}","wb")
        dump(i0,file); file.close()
    
    @staticmethod
    def export_backorders(weights, seed_ix, backo, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Backorders/Backorders_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Backorders_{weights}_{seed_ix}","wb")
        dump(backo,file); file.close()
    
    @staticmethod
    def export_perished(weights, seed_ix, perished, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Perished/Perished_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Perished_{weights}_{seed_ix}","wb")
        dump(perished,file); file.close()
    
    @staticmethod
    def export_norm_matrix(weights, seed_ix, norm_matrix, other_path=False, theta=1.0, gamma=1.0, min_q=0.1):

        path = "C:/Users/ari_r/" if not other_path else "C:/Users/a.rojasa55/"
        new_dir = path + f"OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Service_Level_{theta}/Gamma_{gamma}/Min_q{min_q}/Matrix/Matrix_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Matrix_{weights}_{seed_ix}","wb")
        dump(norm_matrix,file); file.close()


class import_results():
