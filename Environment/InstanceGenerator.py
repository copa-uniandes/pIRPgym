################################## Modules ##################################
### Basic Librarires
import numpy as np; from copy import copy, deepcopy; import matplotlib.pyplot as plt
import networkx as nx; import sys; import pandas as pd; import math; import numpy as np
import time; from termcolor import colored
from random import random, seed, randint, shuffle

### Optimizer
import gurobipy as gu

### Renderizing
import imageio

### Gym & OR-Gym
import gym; #from gym import spaces
from or_gym import utils


class instance_generator():

    def __init__(self, r_seed = 0, **kwargs):
        seed(r_seed)
        
        ### Main parameters ###
        self.M = 10                                    # Suppliers
        self.K = 10                                    # Products
        self.F = 4                                     # Fleet
        self.T = 7

        self.sprice = (1,500)
        self.hprice = (1,500)

        self.lambda1 = 0.5

        self.Q = 1000 # TODO !!!!!!!!
    
        self.S = 4              # Number of sample paths
        self.LA_horizon = 5     # Look-ahead time window's size
        
        ### historical log parameters ###
        self.hist_window = 40       # historical window

        utils.assign_env_config(self, kwargs)


    def gen_availabilities(self):

        
        pass

    
    