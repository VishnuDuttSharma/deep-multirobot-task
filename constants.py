import numpy as np

NUM_ROBOT = 2 #number of robots
REWARD_THRESH = 0.80 # mimimum value of a reward
PATH_LEN = 2 # Path length/Time horizon
HEIGHT = 16 # Heigh of the grid
WIDTH = 16 # Width of the grid
STEP = 3
FOV = 3
NUM_GNN_FEAT = 15

# Dictionary to save directions
DIR_DICT = {
    0: np.array([-STEP,0]), #up
    1: np.array([ STEP,0]), # down
    2: np.array([0,-STEP]), # left
    3: np.array([0, STEP]) # right
}

# List containing numpy array orresponding to the directions
DIR_LIST = np.array(list(DIR_DICT.values()))

