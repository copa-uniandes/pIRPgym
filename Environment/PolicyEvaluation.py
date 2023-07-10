from InstanceGenerator import instance_generator
from SD_IB_IRP_PPenv import steroid_IRP
from Policies import policy_generator
from time import process_time


def Policy_evaluation(inst_gen:instance_generator,env:steroid_IRP):  
    # Episode's and performance storage
    rewards = {};   states = {};   real_actions = {};   backorders = {};   la_decisions = {}
    perished = {}; actions={}; #times = {}

    run_time = process_time()

    state = env.reset(inst_gen,return_state=True)
    
    done = False
    while not done:
        #print_state(env)
        # Environment transition
        states[env.t] = state

        # Transition
        #print(f"Day {env.t}")
        if inst_gen.other_params["demand_type"] == "aggregated":
            action, la_dec = policy_generator.Inventory.Stochastic_Rolling_Horizon(state,env,inst_gen)

        else:
            action, la_dec = policy_generator.Inventory.Stochastic_RH_Age_Demand(state,env,inst_gen)

        state, reward, done, real_action, _,  = env.step(action[1:],inst_gen)
        if done:   states[env.t] = state
        
        # Data storage
        actions[env.t-1] = action
        real_actions[env.t-1] = real_action
        backorders[env.t-1] = _["backorders"]
        perished[env.t-1] = {k:_["perished"][k] if k in _["perished"] else 0 for k in inst_gen.Products}
        rewards[env.t] = reward
        la_decisions[env.t-1] = la_dec
    
    #times = time.time() - run_time

    return [rewards, states, real_actions, backorders, la_decisions, perished, actions]


def run_instance(num_episodes, discount = ("strong","conc"), dem_dist = [2,0.5]):
    
    ''' Fixed Parameters '''
    
    backorders = 'backorders'                                       # Feature's parameters
    stochastic_params = ['q','d']

    look_ahead = ['q','d']
    historical_data = ['*']

    env_config = { 'M': 5, 'K': 10, 'T': 7,  'F': 4,
                'S': 10,  'LA_horizon': 4, 'back_o_cost': 2000}      # Other parameters

    q_params = {'dist': 'c_uniform', 'r_f_params': [6,20]}          # Offer
    p_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}

    h_params = {'dist': 'd_uniform', 'r_f_params': [20,61]}         # Holding costs

    ''' Demand Distribution and Price Discount '''

    d_params = {'dist': 'log-normal', 'r_f_params': dem_dist}
    disc = discount

    demand_type = "age"
    I0 = 5

    stoch_rd_seed = 0                                               # Random seeds
    det_rd_seed = 1

    policy1 = {}; policy2 = {}
    ep = 0
    det_rd_seed = randint(0,int(2e7))
    while ep < num_episodes:
        stoch_rd_seed = randint(0,int(2e7))

        inst_gen = instance_generator(look_ahead, stochastic_params, historical_data, backorders, demand_type, env_config = env_config)
        inst_gen.generate_random_instance(det_rd_seed, stoch_rd_seed, I0, q_params = q_params, p_params = p_params, d_params = d_params, h_params = h_params, discount = disc)

        policy1[ep] = Policy_evaluation(inst_gen) + [inst_gen]
        #policy2[ep] = Policy_evaluation(inst_gen) + [inst_gen]
        print(f"Done episode {ep}")
        ep += 1
    
    return policy1, policy2



















