from ...Blocks.InstanceGenerator import instance_generator
from ...Blocks.pIRPenv import steroid_IRP
from ...Blocks import Policies
from time import process_time

def single_episode_evaluation(inst_gen,env,routing_policy):
    ''' Simulations '''
    num_episodes = 1


    print(f'################### Episode simulation ##################')
    # Episode's and performance storage
    rewards=dict();  states=dict();   real_actions=dict();   backorders=dict();   la_decisions=dict()
    perished=dict(); actions=dict(); #times=dict() 

    routing_performance = dict()
    run_time = process_time()

    time_limit = 10

    # Reseting the environment
    state = env.reset(inst_gen,return_state=True)

    done = False
    while not done:
        print(f'-------------------- Step {env.t} --------------------')
        #print_state(env)
        # Environment transition
        states[env.t] = state 

        if inst_gen.other_params["demand_type"] == "aggregated":
            ''' Purchase'''
            [purchase,demand_compliance], la_dec = pIRPgym.Policies.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)

            ''' Routing '''
            nn_routes, nn_distances, nn_loads, nn_time = pIRPgym.Policies.Routing.NearestNeighbor(purchase,inst_gen,env.t)                      # Nearest Neighbor

            [GA_routes,GA_distances,GA_loads,GA_time],GA_top = pIRPgym.Policies.Routing.HybridGenticAlgorithm(purchase,inst_gen,env.t,top=False,rd_seed=0,time_limit=time_limit);print('✅ GA routing')   # Genetic Algorithm
            GA_extra_cost = env.compute_solution_real_cost(inst_gen,GA_routes,purchase)                     

            ''' Compound action'''        
            action = {'routing':GA_routes, 'purchase':purchase, 'demand_compliance':demand_compliance}

        state, reward, done, real_action, _,  = env.step(action,inst_gen)
        if done:   states[env.t] = state
        
        # Data storage
        actions[env.t-1] = action
        real_actions[env.t-1] = real_action
        backorders[env.t-1] = _["backorders"]
        perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
        # rewards[env.t] = reward
        la_decisions[env.t-1] = la_dec

    print('Finished')