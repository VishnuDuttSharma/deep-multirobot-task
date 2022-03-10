import numpy as np

NUM_ROBOT = 20 #number of robots
REWARD_THRESH = 0.975 #0.90 #0.975 # mimimum value of a reward
PATH_LEN = 2 # Path length/Time horizon
HEIGHT = int(np.sqrt(NUM_ROBOT*(100*100/20))) # int(100*NUM_ROBOT/20) # Height of the grid
WIDTH = int(np.sqrt(NUM_ROBOT*(100*100/20))) # int(100*NUM_ROBOT/20) # Width of the grid
GRID_SIZE = HEIGHT
STEP = 20 # 8 #15 #8# 12 #20 #8 #20 #8 #20 #8 # 6
FOV = 6 #3
NUM_TGT_FEAT = 20 #40
NUM_ROB_FEAT = 10
COMM_RANGE =  10. #15 #20 #10. #20.
# SAMPLE_SIZE = 1000

# Dictionary to save directions
DIR_DICT = {
    0: np.array([0, 0]), # Stay
    1: np.array([-STEP,0]), #up
    2: np.array([ STEP,0]), # down
    3: np.array([0,-STEP]), # left
    4: np.array([0, STEP]) # right
}

# List containing numpy array orresponding to the directions
DIR_LIST = np.array(list(DIR_DICT.values()))
