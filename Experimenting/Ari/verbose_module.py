import numpy as np
from pickle import dump, load
import os

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
        
        inventory, purchase, backorders = la_decisions[0], la_decisions[1], la_decisions[2]; impact = dict()

        purch_cost = sum(inst_gen.W_p[env.t][i,k]*purchase[t][s][i,k] for t in purchase for s in inst_gen.Samples for k in inst_gen.Products for i in inst_gen.M_kt[k,env.t + t])/len(inst_gen.Samples)
        backo_cost = sum(inst_gen.back_o_cost[k]*backorders[t][s][k] for t in backorders for s in inst_gen.Samples for k in inst_gen.Products)/len(inst_gen.Samples)
        w = {(i,t,s):1 if sum(purchase[t][s][i,k] for k in inst_gen.Products if i in inst_gen.M_kt[k,env.t+t])>0 else 0 for t in purchase for s in inst_gen.Samples for i in inst_gen.Suppliers}
        rout_aprox_cost = sum((inst_gen.c[0,i]+inst_gen.c[i,0])*w[i,t,s] for (i,t,s) in w)/len(inst_gen.Samples)
        holding_cost = sum(inst_gen.W_h[t][k]*inventory[t][s][k,o] for t in inventory for s in inst_gen.Samples for k in inst_gen.Products for o in range(inst_gen.O_k[k]))/len(inst_gen.Samples)
        if inst_gen.hold_cost: impact["costs"] = purch_cost + backo_cost + rout_aprox_cost + holding_cost
        else: impact["costs"] = purch_cost + backo_cost + rout_aprox_cost
        
        for e in inst_gen.E:
            transport = sum((inst_gen.c_LCA[e][k][0,i]+inst_gen.c_LCA[e][k][i,0])*purchase[t][s][i,k] for t in purchase for s in inst_gen.Samples for k in inst_gen.Products for i in inst_gen.M_kt[k,env.t+t])/len(inst_gen.Samples)
            storage = sum(inst_gen.h_LCA[e][k]*inventory[t][s][k,o] for t in inventory for s in inst_gen.Samples for k in inst_gen.Products for o in range(inst_gen.O_k[k]))/len(inst_gen.Samples)
            waste = sum(inst_gen.waste_LCA[e][k]*inventory[t][s][k,inst_gen.O_k[k]] for k in inst_gen.Products for t in inventory for s in inst_gen.Samples)/len(inst_gen.Samples)

            impact[e] = transport + storage + waste
            #impact[e] = transport + waste

        print("\n\t"+" "*8,*[" "*int(np.floor((15-len(e))/2))+e+" "*int(np.ceil((15-len(e))/2))+"\t" for e in ["costs"]+inst_gen.E],sep="\t")
        print("\tresults ",*[f"{impact[e]:.2e}" + f" ({(impact[e]-env.norm_matrix[e]['best'])/(env.norm_matrix[e]['worst']-env.norm_matrix[e]['best']):.2f}) " for e in ["costs"]+inst_gen.E],sep="\t")


class export_results():

    @staticmethod
    def export_rewards(weights,seed_ix,rewards, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Rewards/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Rewards/"
        new_dir = path + f"Rewards_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Rewards_{weights}_{seed_ix}","wb")
        dump(rewards,file); file.close()
    
    @staticmethod
    def export_actions(weights,seed_ix,action, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Actions/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Actions/"
        new_dir = path + f"Actions_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Actions_{weights}_{seed_ix}","wb")
        dump(action,file); file.close()
    
    @staticmethod
    def export_lookahead_decisions(weights,seed_ix,lookahead, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Lookahead/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Lookahead/"
        new_dir = path + f"Lookahead_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Lookahead_{weights}_{seed_ix}","wb")
        dump(lookahead,file); file.close()
    
    @staticmethod
    def export_instance_parameters(weights,seed_ix,inst_gen, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Instance/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Instance/"
        new_dir = path + f"Instance_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Instance_{weights}_{seed_ix}","wb")
        dump(inst_gen,file); file.close()

    @staticmethod
    def export_inventory(weights, seed_ix, i0, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Inventory/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Inventory/"
        new_dir = path + f"Inventory_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Inventory_{weights}_{seed_ix}","wb")
        dump(i0,file); file.close()
    
    @staticmethod
    def export_backorders(weights, seed_ix, backo, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Backorders/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Backorders/"
        new_dir = path + f"Backorders_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Backorders_{weights}_{seed_ix}","wb")
        dump(backo,file); file.close()
    
    @staticmethod
    def export_perished(weights, seed_ix, perished, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Perished/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Perished/"
        new_dir = path + f"Perished_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Perished_{weights}_{seed_ix}","wb")
        dump(perished,file); file.close()
    
    @staticmethod
    def export_norm_matrix(weights, seed_ix, norm_matrix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Matrix/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Matrix/"
        new_dir = path + f"Matrix_{weights}/"

        if not os.path.exists(new_dir): os.makedirs(new_dir)

        file = open(new_dir+f"Matrix_{weights}_{seed_ix}","wb")
        dump(norm_matrix,file); file.close()

class import_results():

    @staticmethod
    def import_rewards(weights,seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Rewards/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Rewards/"
        new_dir = path + f"Rewards_{weights}/"

        file = open(new_dir+f"Rewards_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp
    
    @staticmethod
    def import_actions(weights,seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Actions/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Actions/"
        new_dir = path + f"Actions_{weights}/"

        file = open(new_dir+f"Actions_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp
    
    @staticmethod
    def import_lookahead_decisions(weights,seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Lookahead/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Lookahead/"
        new_dir = path + f"Lookahead_{weights}/"

        file = open(new_dir+f"Lookahead_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp
    
    @staticmethod
    def import_instance_parameters(weights,seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Instance/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Instance/"
        new_dir = path + f"Instance_{weights}/"

        file = open(new_dir+f"Instance_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp

    @staticmethod
    def import_inventory(weights, seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Inventory/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Inventory/"
        new_dir = path + f"Inventory_{weights}/"

        file = open(new_dir+f"Inventory_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp
    
    @staticmethod
    def import_backorders(weights, seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Backorders/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Backorders/"
        new_dir = path + f"Backorders_{weights}/"

        file = open(new_dir+f"Backorders_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp
    
    @staticmethod
    def import_perished(weights, seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Perished/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Perished/"
        new_dir = path + f"Perished_{weights}/"

        file = open(new_dir+f"Perished_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp

    @staticmethod
    def import_norm_matrix(weights, seed_ix, other_path=False):

        path = "C:/Users/ari_r/OneDrive - Universidad de los Andes/1. MIIND/Tesis/Experimentos/Matrix/"
        if other_path: path = "C:/Users/a.rojasa55/OneDrive - Universidad de los andes/1. MIIND/Tesis/Experimentos/Matrix/"
        new_dir = path + f"Matrix_{weights}/"

        file = open(new_dir+f"Matrix_{weights}_{seed_ix}","rb")
        resp = load(file); file.close()

        return resp