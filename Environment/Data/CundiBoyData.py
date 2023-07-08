#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_suppliers = pd.read_excel('Data_Fruver_0507.xlsx', sheet_name='provider_orders')
data_demand = pd.read_excel("./Data_Fruver_0507.xlsx",sheet_name="daily_sales_historic")
data_demand = data_demand[["date","store_product_id","sales"]]


#%% Demand
K = list(pd.unique(data_demand["store_product_id"]))

data_demand = data_demand.groupby(by=["date","store_product_id"],as_index=False).sum()
hist_demand = {k:[data_demand.loc[i,"sales"] for i in data_demand.index if data_demand.loc[i,"store_product_id"] == k and data_demand.loc[i,"sales"] > 0] for k in K}
total_sales = pd.DataFrame.from_dict({k:sum(hist_demand[k]) for k in K},orient="index")
df_sorted = total_sales.sort_values(by=0, ascending=False)

# Retrieve the top 30 products with the highest sales
K = list(df_sorted.head(30).index)
hist_demand = {k:hist_demand[k] for k in K}


#%% Subset of suppliers & expected demand
M = list()
K_pur = list()

M_k = dict()
K_i = dict()

ordered = dict()
delivered = dict()

d = dict()

for obs in data_suppliers.index:
    i = data_suppliers['provider_id'][obs]
    k = data_suppliers['store_product_id'][obs]

    if i not in M:
        M.append(i)
        K_i[i] = list()
    
    if k not in K_pur:
        K_pur.append(k)
        M_k[k] = list()
        d[k] = 0

    M_k[k].append(i)
    K_i[i].append(k)

    if (i,k) not in ordered.keys():
        ordered[i,k] = 0
        delivered[i,k] = 0
    
    ordered[i,k] += data_suppliers['quantity_order'][obs]
    delivered[i,k] += data_suppliers['quantity_received'][obs]

for i in M:
    K_i[i] = set(K_i[i])
    K_i[i] = list(K_i[i])

for k in K_pur:
    M_k[k] = set(M_k[k])
    M_k[k] = list(M_k[k])

service_level = dict()
for (i,k) in ordered.keys():
    service_level[i,k] = delivered[i,k]/ordered[i,k]


#%% Compute the average demand per product
ex_d_per_day = {}
for k in K:
    ex_d_per_day[k] = sum(hist_demand[k])/321

plt.bar(range(len(K)),list(ex_d_per_day.values()))
plt.show()


#%% Compute availability
q = dict()
for k in K:
    target_demand = ex_d_per_day[k]*1.5
    total_vals = sum([service_level[i,k] for i in M_k[k]])

    for i in M:
        if k in K_i[i]:
            q[i,k] = target_demand*service_level[i,k]/total_vals
        else:
            q[i,k] = 0



# %%
