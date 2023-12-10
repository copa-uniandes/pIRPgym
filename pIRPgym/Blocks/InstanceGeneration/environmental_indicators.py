
import pandas as pd
import os


class indicators():

    @staticmethod
    def get_environmental_indicators(inst_gen):

        K = inst_gen.Products[:min(len(inst_gen.Products),7)]
        script_dir = os.path.dirname(os.path.abspath(__file__)); file = f"{script_dir}/LCA indicators.xlsx"

        df = pd.read_excel(io=file, sheet_name="Transport", index_col=[0])
        c_LCA = {e:{k:{(i,j):inst_gen.c[i,j]*df.loc[e,k] for i in inst_gen.V for j in inst_gen.V if i!= j} for k in K} for e in inst_gen.E}

        df = pd.read_excel(io=file, sheet_name="Storage", index_col=[0])
        h_LCA = {e:{k:df.loc[e,k] for k in K} for e in inst_gen.E}

        df = pd.read_excel(io=file, sheet_name="Waste", index_col=[0])
        waste_LCA = {e:{k:df.loc[e,k] for k in K} for e in inst_gen.E}

        return c_LCA, h_LCA, waste_LCA
