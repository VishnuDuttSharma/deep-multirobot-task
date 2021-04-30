import sys
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import time
from itertools import product

from constants import *

np.random.seed(1232)

  
def calculate_reward(grid, robot_pos, action_list, fov=FOV, get_mask=False):
    """
    Function to calculate the reward calculated by all the robots based on an action vector.
    For this we first update locations of each robot, then create a mask which has 1s only around the new robots locations (square of side (2*FOV+1) for each robot) 
    
    Parameters
    ----------
        grid: 2D grid containing rewards
        robot_pos: Current position for each robot on the grid (NUM_ROBOTx2 size vector)
        action_list: List of action for each robot
    
    Returns
    -------
        total_reward: Total reward calculated by the robots using action_list (the action vector)
    """
    # Convert the integer actions to 2D vector of location differences using DIR_DICT dictionary
    act = np.array([DIR_DICT[k] for k in action_list])
    # Calcuate new locations for each robot
    new_pos = robot_pos + act
    # Make sure that the new locatiosn are within the grid 
    new_pos = new_pos.clip(min=0, max=GRID_SIZE-1)
    
    # Initialize a mask of same shape as grid
    mask = np.zeros(grid.shape, dtype=int)
    
    # iterate over each robot position
    for c_pos, n_pos in zip(robot_pos, new_pos):
        # Set the values to 1 in the mask at each robot's fov
        # also make sure that the indices do not go out of grid
        
        # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
        # This box has a padding of size FOV on each size
        r_lim_lef = max(0, min(c_pos[0]-fov, n_pos[0]-fov))
        c_lim_top = max(0, min(c_pos[1]-fov, n_pos[1]-fov))
        r_lim_rgt = min(max(c_pos[0]+fov+1, n_pos[0]+fov+1), GRID_SIZE)
        c_lim_bot = min(max(c_pos[1]+fov+1, n_pos[1]+fov+1), GRID_SIZE)
        
        # Set the locations withing mask (i.e. witing robot's vision when it moved) to 1
        mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1
        
    if get_mask:
        return mask
        
    # Find total reward as number of 1s in the masked grid
    total_reward = np.sum(grid * mask)
    
    return total_reward


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--niter', type=int, default=10)
    argparser.add_argument('--save_path', type=str, default=None)
     

    args = argparser.parse_args()

    print(f'#Iterations: {args.niter}')
    print(f'Data Path: {args.save_path}')

    

    # for i in tqdm(range(100)):
    #     pickle.dump(generate_data(args.batch_size), open(f'{args.save_path}/data_{i+1}.pkl', 'wb'))

    
    grid_data = pickle.load(open(f'{args.save_path}/grid_data.pkl', 'rb'))
    robot_pos_data = pickle.load(open(f'{args.save_path}/robot_pos_data.pkl', 'rb'))
    
    perm = [[1,2,3,4] for i in range(NUM_ROBOT)]
    
    
    
    time_list = []
    reward_list = []
    action_list = []

    for i in range(args.niter):
        print(f'Iteration: {i}')
        '''
        tmp_reward_list = []
        tmp_action_list = []
        '''
        max_rwd = -1
        best_id = -1
        best_act = None
        comb = product(*perm)
        
        start_time = time.time()
        for k, c in tqdm(enumerate(comb)):
            rwd = calculate_reward(grid_data[i], robot_pos=robot_pos_data[i], action_list=list(c), fov=FOV, get_mask=False)
            '''
            tmp_reward_list.append(rwd)
            # print(tmp_reward_list)
            tmp_action_list.append(c)
            '''
            if rwd > max_rwd:
                max_rwd = rwd
                best_act = c
        
            #if k == 5:
            #    break
        
        '''
        idx = np.argmax(tmp_reward_list)
        # print(idx)
        reward_list.append(tmp_reward_list[idx])
        action_list.append(tmp_action_list[idx])
        '''
        
        reward_list.append(max_rwd)
        action_list.append(best_act)
        
        time_list.append(time.time() - start_time)
        
    pickle.dump(reward_list, open(f'{args.save_path}/bfs_reward.pkl', 'wb'))
    pickle.dump(action_list, open(f'{args.save_path}/bfs_action.pkl', 'wb'))
    pickle.dump(time_list, open(f'{args.save_path}/bfs_time.pkl', 'wb'))

    print('Done!')