# SD-IB-IRP-PP
A compilation of OR tools for the Stochastic-Dynamic Inventory-Routing-Problem with Perishable Products.

## Powelleskian model

### State ($S_t$)
-   **Physical State** ($R_t$):

        state:  Current available inventory: (dict)  Inventory of product k \in K of age o \in O_k. Since the state is the inventory before the decision is taken, there is no age 0 inventory.
                
        When backlogs are activated, will appear under age 'B'   
                                           

-   **Other deterministic info** ($Z_t$):

        p: Prices: (dict) Price of product k \in K at supplier i \in M
    
        q: Available quantities: (dict) Available quantity of product k \in K at supplier i \in M
    
        h: Holding cost: (dict) Holding cost of product k \in K
    
        historical_data: (dict) Historical log of information (optional)
    
-   **Belief State** ($B_t$):
    
        sample_paths: Simulated sample paths (optional)


### Actions ($X_t$)
The action can be seen as a three level-decision. These are the three layers:

1. Routes to visit the selected suppliers

2. Quantities to purchase from each supplier

3. Demand compliance plan, dispatch decision

4. (Optional) Backlogs compliance

Accordingly, the action will be a list composed as follows:

$$ X = [\text{routes, purchase, demand compliance, backlogs compliance}] $$

        routes: (list) list of lists, each with the nodes visited on the route (including departure and arriving to the depot)

        purchase: (dict) Units to purchase of product k \in K at supplier i \in M

        demand_compliance: (dict) Units of product k in K of age o \in O_k used to satisfy the demand

        backlogs_compliance: (dict) Units of product k in K of age o \in O_k used to satisfy the backlogs

### Exogenous information ($W_t$)!!!!!!


### Transition Function ($S_M^X$)

### Cost Function ($C_t$)
