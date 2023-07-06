#%%

import pandas as pd
import numpy as np

#%% Demand
data_demand = pd.read_excel("./Data_Fruver_0507.xlsx",sheet_name="daily_sales_historic")
data_demand = data_demand[["date","store_product_id","sales"]]
K = list(pd.unique(data_demand["store_product_id"]))

data_demand = data_demand.groupby(by=["date","store_product_id"],as_index=False).sum()
hist_demand = {k:[data_demand.loc[i,"sales"] for i in data_demand.index if data_demand.loc[i,"store_product_id"] == k and data_demand.loc[i,"sales"] > 0] for k in K}
total_sales = pd.DataFrame.from_dict({k:sum(hist_demand[k]) for k in K},orient="index")
df_sorted = total_sales.sort_values(by=0, ascending=False)

# Retrieve the top 30 products with the highest sales
K = list(df_sorted.head(30).index)
hist_demand = {k:hist_demand[k] for k in K}

