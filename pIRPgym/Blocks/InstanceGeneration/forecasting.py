from numpy.random import random,randint


# Auxuliary sample value generator function
def empiric_distribution_sampling(hist,seed=None):
    ''' 
    Sample value generator function.
    Returns a generated random number using acceptance-rejection method.
    Parameters:
    - hist: (list) historical dataset that is used as an empirical distribution for
            the random number generation
    '''
    if seed != None:
        seed(randint(0,int(1e6)))
    Te = len(hist)
    sorted_data = sorted(hist)
    
    prob, value = [], []
    for t in range(Te):
        prob.append((t+1)/Te)
        value.append(sorted_data[t])
    
    # Generates uniform random value for acceptance-rejection testing
    U = random()
    # Tests if the uniform random falls under the empirical distribution
    test = [i>U for i in prob]  
    # Takes the first accepted value
    sample = value[test.index(True)]
    
    return sample