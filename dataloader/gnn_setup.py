import sys
import numpy as np
import argparse
from tqdm import tqdm
import pickle

from dataloader.constants import *
# from constants import *

np.random.seed(1232)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_adjacency_matrix(robot_pos, comm_range):
    '''
    Function to get the adjacency matrix )normalized by max eigen value)

    Parameters
    ----------
        robot_pos: Current locations for the robots
        comm_range: Maximum communication range
    
    Returns
    -------
        adj_mat: 2D array (symmetric) 
    '''
    # Get the number of robots
    num_rob = len(robot_pos)
    # Create an empty num_robots x num_robots sized matrix
    adj_mat = np.zeros((num_rob, num_rob))

    # Iterate over rows
    for i in range(0, num_rob):
        # Iterate over columns
        for j in range(i+1, num_rob):
            # Set (i,j) and (j,i) entry to distance between robots
            adj_mat[i,j] = adj_mat[j,i] = np.linalg.norm(robot_pos[i] - robot_pos[j])

    # If distance between robots is out of communucation range, set it to 0
    adj_mat[adj_mat > comm_range] = 0.
    # Convert the non-zero values to 1 (distance between robots <= communication range)
    adj_mat = (adj_mat > 0).astype(float)


    # Normalize the adj matrix by its mox eigen value
    max_eig_val = np.real(np.max(np.linalg.eigvals(adj_mat))) # abs to tackle complex numbers
    # if max_eig_val == 0:
    #     print(robot_pos)
    #     print(adj_mat)
    
    adj_mat = adj_mat.astype(float) / max_eig_val
    # print(max_eig_val)
    
    return adj_mat

def get_initial_pose(grid, comm_range):
    """
    Function to generate initial positions for the robots. 
    For this we keep generating random locations on the grid till each robot 
    is connected to atleast 1 more robot

    Parameters
    ----------
        grid: 2D grid containing rewards/targets

    Returns
    -------
        initial_pos: Vector of size NUM_ROBOTx2 containing positions for each robot
    """
    # Find number of robots
    num_robot = NUM_ROBOT

    '''
    # Get indices of those rows and columns which are not-occupied (=0 on grid)
    rows, cols = np.where(grid <= 0)
    # Convert to paired indices
    indices = list(zip(rows, cols))

    # Create empty vector for robot locations
    initial_pos = np.zeros((num_robot, 2), dtype=int)
    # Find candidate indices from the paired index list
    cand_indices = np.random.choice(range(len(indices)), num_robot)
    
    # Save the indices into the vector
    for idx, indc in enumerate(cand_indices):
        initial_pos[idx,:] = np.array(indices[indc])
    '''
    
    degree_lt_1 = True
    is_symm = False
    adj_is_nan = True
    # removing constraint
    # while(degree_lt_1):
    
    while adj_is_nan:
        # Generate random location for each robot
        initial_pos = np.random.randint(low=0, high=grid.shape[0], size=(num_robot, 2))
    
        # get corresponding adjacency matrix
        adj_mat = get_adjacency_matrix(initial_pos, comm_range)
        
        adj_is_nan = np.isnan(adj_mat).any()

    # Check minimum degree. (degree_lt_1 = is any robot connected to 0 robots)
    # degree_lt_1 = ((adj_mat > 0).sum(axis=0) == 0).any()
    # if not degree_lt_1:
    #     print('Generating again')
        
    # Remove the rewards from the grid at robots' locations
    for pos in initial_pos:
        grid[pos[0], pos[1]] = 0

    return initial_pos, adj_mat

def get_reward_grid(height, width, reward_thresh=REWARD_THRESH):
    """
    Generate reward grid/environment to be explores
    Parameters
    ----------
        height: Height of the grid
        width: Width of the grid
    Returns
    -------
        grid containign reward
    """
    # Random grid (10 times the actuial size)
    reward_grid = np.random.randint(1,100, size=(10*height, 10*width)) / 100.
    # downsample to the original size
    reward_grid_orig = reward_grid[::10, ::10]

    # copying , not necessary. doesn't affect
    reward_grid = reward_grid_orig.copy()

    # Keep only those values which are above reward_thresh. Helps in making the rewards sparse
    mask = reward_grid > reward_thresh

    return mask.astype(int)
  
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
    grid_size = grid.shape[0]
    # Convert the integer actions to 2D vector of location differences using DIR_DICT dictionary
    act = np.array([DIR_DICT[k] for k in action_list])
    # Calcuate new locations for each robot
    new_pos = robot_pos + act
    # Make sure that the new locatiosn are within the grid 
    new_pos = new_pos.clip(min=0, max=grid_size-1)
    
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
        r_lim_rgt = min(max(c_pos[0]+fov+1, n_pos[0]+fov+1), grid_size)
        c_lim_bot = min(max(c_pos[1]+fov+1, n_pos[1]+fov+1), grid_size)
        
        # Set the locations withing mask (i.e. witing robot's vision when it moved) to 1
        mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1
        
    if get_mask:
        return mask
        
    # Find total reward as number of 1s in the masked grid
    total_reward = np.sum(grid * mask)
    
    return total_reward

def greedy_action_finder(grid, robot_pos, fov):
    '''
    Function to greedily find actions for all robot.
    For a robot, find an action which results in covring most number of targets.
    Remove the targets from the grid.
    Repeat for next robot.

    Parameters
    ----------
        grid: 2D grid containing rewards
        robot_pos: Current position for each robot on the grid (NUM_ROBOTx2 size vector)
        fov: Field of View in each direction. FOV=2 results in 5x5 grid centered at robot location
        
    Returns
    -------
        robot_acts: List of actions for each robot
        reward: Reward over the greedy 
    '''
    grid_size = grid.shape[0]
    # Copy teh original grid for later use
    orig_grid = grid.copy()
    # List of actions
    act_list = ['Stay', 'Up', 'Down', 'Left', 'Right']
    # List to save the actiosn for each robot
    robot_acts = []
    # Array to save the robot locations
    new_pos = np.zeros(robot_pos.shape, dtype=robot_pos.dtype)

    # Iterate over each robot
    for i_rob in range(len(robot_pos)):
        # List to save reward for each action
        reward_list = []
        
        # iterate over each action
        for i_act, act in enumerate(act_list):
            # Copy the grid to avoid overwriting
            test_grid = grid.copy()
            # Move robot to new location according to the action
            r_pos = robot_pos[i_rob] + DIR_DICT[i_act]
            # Keep the robot within the grid by limiting x-y coordinated in range [9,grid size)
            r_pos = r_pos.clip(min=0, max=grid_size-1)

            # Find reward. It is the number of total targets in robot's FOV
            reward = np.sum(test_grid[max(0,r_pos[0]-fov):min(r_pos[0]+FOV+1,grid_size),
                                max(0,r_pos[1]-fov):min(r_pos[1]+FOV+1,grid_size)])
            # Add the reward to teh list
            reward_list.append(reward)
        
        # Find the best action = argmax(reward list)
        best_act = np.argmax(reward_list)
        # Save this action as the Robot's action as per greedy algorithm
        robot_acts.append(best_act)
        
        # Apply action to the currect robot. Move to new location
        r_pos = robot_pos[i_rob] + DIR_DICT[best_act]
        # Save the new location
        new_pos[i_rob] = r_pos

        # Remove the targets within Robot's field of view
        test_grid[max(0,r_pos[0]-FOV):min(r_pos[0]+fov+1,grid_size),
                    max(0,r_pos[1]-FOV):min(r_pos[1]+fov+1,grid_size)] = 0
        
        # Update the grid
        grid = test_grid.copy()

    # Find the reward of the actions
    reward = calculate_reward(orig_grid, robot_pos, robot_acts)

    # return robot_acts, new_pos, reward
    return robot_acts, reward

def random_action_finder(grid, robot_pos, sample_size):
    '''
    Function to randomly sample action vectors (an action for each robot) and return the best one

    Parameters
    ----------
        grid: 2D Grid
        robot_pos: Current location of all the robots (NUM_ROBOTx2 size vector)
        sample_size: Number of action vectors to sample

    Returns
    -------
        Best performing action vector
        Corresponding reward
    '''
    num_robot = robot_pos.shape[0]
    # Generate random actions
    action_space = np.random.randint(low=0,high=len(DIR_DICT.keys()), size=(sample_size, num_robot))
    # List to save rewards for all actions
    reward_list = []

    # Iterate over each action in the space
    for i_samp in range(sample_size):
        # Calculate reward over an action vector
        reward = calculate_reward(grid, robot_pos, action_space[i_samp]) 
        # Save the reward to the list
        reward_list.append(reward)
    
    # Get the index for the highest reward
    best_samp = np.argmax(reward_list) 
    # Return best action vector and the reward
    return action_space[best_samp], reward_list[best_samp]  

def centralized_greedy_action_finder(grid, robot_pos, fov):
    '''
    Function to greedily find actions for all robot.
    For a robot, find an action which results in covring most number of targets.
    Remove the targets from the grid.
    Repeat for next robot.

    Parameters
    ----------
        grid: 2D grid containing rewards
        robot_pos: Current position for each robot on the grid (NUM_ROBOTx2 size vector)
        fov: Field of View in each direction. FOV=2 results in 5x5 grid centered at robot location
        
    Returns
    -------
        robot_acts: List of actions for each robot
        reward: Reward over the greedy 
    '''
    grid_size = grid.shape[0]
    # Get the number of robots
    n_rob = robot_pos.shape[0] #NUM_ROBOT
    # Copy the original grid for later use
    orig_grid = grid.copy()
    # List of actions
    act_list = ['Stay', 'Up', 'Down', 'Left', 'Right']
    # List to save the actions for each robot
    robot_acts = [None]*n_rob
    # Array to save the robot locations
    new_pos = np.zeros(robot_pos.shape, dtype=robot_pos.dtype)

    # List to create track of the robots already taken care of
    visited = [] # S

    # mask to help with calucating the reward
    mask = np.zeros(grid.shape, dtype=int)
    
    for k in range(n_rob):
        # create the mask with visited robots
        if len(visited) > 0:
            # mask the areas covered by the last robot which was added to the set
            last_robot_id = visited[-1] # Last added robot's ID
            
            c_pos = robot_pos[last_robot_id] # current pos
            n_pos = c_pos + DIR_DICT[ robot_acts[last_robot_id] ]
            # Keep the robot within the grid by limiting x-y coordinated in range [0,grid size)
            n_pos = n_pos.clip(min=0, max=grid_size-1)

            # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
            # This box has a padding of size FOV on each size
            r_lim_lef = max(0, min(c_pos[0]-fov, n_pos[0]-fov))
            c_lim_top = max(0, min(c_pos[1]-fov, n_pos[1]-fov))
            r_lim_rgt = min(max(c_pos[0]+fov+1, n_pos[0]+fov+1), grid_size)
            c_lim_bot = min(max(c_pos[1]+fov+1, n_pos[1]+fov+1), grid_size)

            # Set the locations withing mask (i.e. witing robot's vision when it moved) to 1
            mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1

        # matrix to save the f values, size: NUM_ROBOT x NUM_ACTIONS
        f_mat = -1*np.ones((n_rob, len(act_list)))

        for i_rob in range(n_rob):
            # If robot already visited 
            if i_rob in visited:
                continue
            
            # List to track rewards for each action
            temp_reward_list = []

            for i_act, act in enumerate(act_list):
                # Copy the mask to avoid overwriting
                temp_mask = mask.copy()
                # copy current pos
                c_pos = robot_pos[i_rob]
                # Move robot to new location according to the action
                n_pos = c_pos + DIR_DICT[i_act]
                # Keep the robot within the grid by limiting x-y coordinated in range [0,grid size)
                n_pos = n_pos.clip(min=0, max=grid_size-1)

                # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
                # This box has a padding of size FOV on each size
                r_lim_lef = max(0, min(c_pos[0]-fov, n_pos[0]-fov))
                c_lim_top = max(0, min(c_pos[1]-fov, n_pos[1]-fov))
                r_lim_rgt = min(max(c_pos[0]+fov+1, n_pos[0]+fov+1), grid_size)
                c_lim_bot = min(max(c_pos[1]+fov+1, n_pos[1]+fov+1), grid_size)

                # Set the locations withing mask (i.e. witing robot's vision when it moved) to 1
                temp_mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1

                # Find reward. It is the number of total targets in robot's FOV
                reward = np.sum(grid * temp_mask)
                # Add the reward to teh list
                temp_reward_list.append(reward)

                # Save it to f
                f_mat[i_rob, i_act] = reward
            
            '''
            # Find the best action and reward
            best_act = np.argmax(temp_reward_list)
            best_rwd = temp_reward_list[best_act]

            # Save it to f
            f_mat[i_rob, best_act] = best_rwd
            '''
            # initialiize f with -1, and update for all actions for the rotbos (move f up in th loop)

        # find which robot provides best reward
        best_rob, best_act = np.where(f_mat == np.max(f_mat))
        # if multiple robots with same rewards exist, pick first of them not alread
        if len(best_rob):
            best_rob = best_rob[0]
            best_act = best_act[0]

        # Add robot to visited list at the end
        visited.append(best_rob)
        # Add the corresponding action to the output list
        robot_acts[best_rob] = best_act

    # Testing: copy the old value of reeward
    old_rwd = np.max(temp_reward_list)
    # Find the reward of the actions (use it as a test here, check with the last best reward)
    reward = calculate_reward(orig_grid, robot_pos, robot_acts)
    # check if they are same (Must be)
    assert reward == old_rwd

    # return robot_acts, new_pos, reward
    return robot_acts, reward


def get_features(grid, robot_pos, fov=FOV, step=STEP, target_feat_size=10, robot_feat_size=10, comm_range=COMM_RANGE):
    '''
    Function to get the features (local position of robot in fov) 

    Parameters
    ----------
        grid: 2D grid containing rewards
        robot_pos: Current position for each robot on the grid (NUM_ROBOTx2 size vector)
        fov: Field of View in each direction. FoV=2 results in 5x5 grid centered at robot location
        step: Step size of the robots
        target_feat_size: For each robot, maximum number of target in FoV to be considered in the feature vector 
        robot_feat_size: For each robot, Maximum number of robots in FoV to be considered in the feature vector
        
    Returns
    -------
        feat_vec: Feature vector containing location of targets and robots in local FoV of each robot. Size: num_robot x (target_feat_size + robot_feat_size)
    '''
    grid_size = grid.shape[0]
    # Get number of robots
    num_rob = robot_pos.shape[0]
    # Create an empty vector for features. size: N_Robot x (targets + robot) x 2
    feat_vec = -1*np.ones((num_rob, target_feat_size + robot_feat_size, 2))

    # Iterate over each robot
    for i_rob in range(num_rob):
        # copy current pos
        c_pos = robot_pos[i_rob]
        
        # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
        # This box has a padding of size FOV on each size. We add step to consider effect of motion in all directions
        r_lim_lef = max(0, c_pos[0]-fov-step)
        c_lim_top = max(0, c_pos[1]-fov-step)
        r_lim_rgt = min(c_pos[0]+fov+step+1, grid_size)
        c_lim_bot = min(c_pos[1]+fov+step+1, grid_size)

        # create the mask with 1s in robot's FOV
        mask = np.zeros(grid.shape)
        mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1

        # Get locatiosn where a target is present
        rows, cols = np.where(mask*grid > 0)
        # get relative position and normalize. (.T returns the tranpose of the matrix)
        rel_pos = np.array([rows - c_pos[0], cols - c_pos[1]]).T / (fov+step)
        # Get the sorting indices (lowest to highest). Sort based on on relative distance
        indices = np.argsort(np.linalg.norm(rel_pos, axis=1))

        # Save the relative normalized locations of the targets in feature vector.
        feat_vec[i_rob, 0:min(target_feat_size, len(indices)), :] = rel_pos[indices][0:min(target_feat_size, len(indices))]

        ### For robots
        # Get relative location on all robots
        rel_pos = robot_pos - c_pos
        # Get the subset containing only those robots which are within robot's FOV. Also normalize them
        # rel_pos_subset =  rel_pos[ (np.abs(rel_pos[:,0]) <= (fov+step)) & (np.abs(rel_pos[:,1]) <= (fov+step))] / (fov+step)
        rel_pos_subset = rel_pos[np.linalg.norm(rel_pos, axis=1) <= comm_range] / float(comm_range)
        # Get the sorting indices (lowest to highest). Sort based on on relative distance
        indices = np.argsort(np.linalg.norm(rel_pos_subset, axis=1))
        # First elemnt (index=0) is the robot it self. Thus remove it from the list
        rel_pos = rel_pos_subset[indices] #rel_pos[indices]
        rel_pos = rel_pos[1:]
        indices = indices[1:]
        
        # Save into the feature vector
        feat_vec[i_rob, target_feat_size:target_feat_size+min(robot_feat_size, len(indices)), :] = rel_pos[:min(robot_feat_size, len(indices))]
        # 20 + 10, 2, 60
    return feat_vec


def get_rect_features(grid, robot_pos, fov=FOV, step=STEP, target_feat_size=10, robot_feat_size=10, comm_range=COMM_RANGE):
    '''
    Function to get the features (local position of robot in fov) using only traversible paarts of the environement (center not included)

    Parameters
    ----------
        grid: 2D grid containing rewards
        robot_pos: Current position for each robot on the grid (NUM_ROBOTx2 size vector)
        fov: Field of View in each direction. FoV=2 results in 5x5 grid centered at robot location
        step: Step size of the robots
        target_feat_size: For each robot, maximum number of target in FoV to be considered in the feature vector 
        robot_feat_size: For each robot, Maximum number of robots in FoV to be considered in the feature vector
        
    Returns
    -------
        feat_vec: Feature vector containing location of targets and robots in local FoV of each robot. Size: num_robot x (target_feat_size + robot_feat_size)
    '''
    grid_size = grid.shape[0]
    # Get number of robots
    num_rob = robot_pos.shape[0]
    # Create an empty vector for features. size: N_Robot x (targets + robot) x 2
    feat_vec = -1*np.ones((num_rob, target_feat_size + robot_feat_size, 2))

    # Iterate over each robot
    for i_rob in range(num_rob):
        # copy current pos
        c_pos = robot_pos[i_rob]
        
        
        
        mask = np.zeros(grid.shape)
        #### Horizontal mask ######
        # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
        # This box has a padding of size FOV on each size. We add step to consider effect of motion in all directions
        r_lim_lef = max(0, c_pos[0]-fov)
        c_lim_top = max(0, c_pos[1]-fov-step)
        r_lim_rgt = min(c_pos[0]+fov+1, grid_size)
        c_lim_bot = min(c_pos[1]+fov+step+1, grid_size)

        # create the mask with 1s in robot's FOV
        mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1
        
        #### Vertical mask ######
        # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
        # This box has a padding of size FOV on each size. We add step to consider effect of motion in all directions
        r_lim_lef = max(0, c_pos[0]-fov-step)
        c_lim_top = max(0, c_pos[1]-fov)
        r_lim_rgt = min(c_pos[0]+fov+step+1, grid_size)
        c_lim_bot = min(c_pos[1]+fov+1, grid_size)

        # create the mask with 1s in robot's FOV
        mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 1
        
        #### Center mask remove ######
        # Calculate the bounding box ranges for the box generated by robot moving from the current location (c_pos) to new location (n_pos)
        # This box has a padding of size FOV on each size. We add step to consider effect of motion in all directions
        r_lim_lef = max(0, c_pos[0]-fov)
        c_lim_top = max(0, c_pos[1]-fov)
        r_lim_rgt = min(c_pos[0]+fov+1, grid_size)
        c_lim_bot = min(c_pos[1]+fov+1, grid_size)

        # create the mask with 1s in robot's FOV
        mask[r_lim_lef:r_lim_rgt, c_lim_top:c_lim_bot] = 0
        
        
        
        
        
        # Get locatiosn where a target is present
        rows, cols = np.where(mask*grid > 0)
        # get relative position and normalize. (.T returns the tranpose of the matrix)
        rel_pos = np.array([rows - c_pos[0], cols - c_pos[1]]).T / (fov+step)
        # Get the sorting indices (lowest to highest). Sort based on on relative distance
        indices = np.argsort(np.linalg.norm(rel_pos, axis=1))

        # Save the relative normalized locations of the targets in feature vector.
        feat_vec[i_rob, 0:min(target_feat_size, len(indices)), :] = rel_pos[indices][0:min(target_feat_size, len(indices))]

        ### For robots
        # Get relative location on all robots
        rel_pos = robot_pos - c_pos
        # Get the subset containing only those robots which are within robot's FOV. Also normalize them
        # rel_pos_subset =  rel_pos[ (np.abs(rel_pos[:,0]) <= (fov+step)) & (np.abs(rel_pos[:,1]) <= (fov+step))] / (fov+step)
        rel_pos_subset = rel_pos[np.linalg.norm(rel_pos, axis=1) <= comm_range] / float(comm_range)
        # Get the sorting indices (lowest to highest). Sort based on on relative distance
        indices = np.argsort(np.linalg.norm(rel_pos_subset, axis=1))
        # First elemnt (index=0) is the robot it self. Thus remove it from the list
        rel_pos = rel_pos_subset[indices] #rel_pos[indices] ##CORRECTED
        rel_pos = rel_pos[1:]
        indices = indices[1:]
        
        # Save into the feature vector
        feat_vec[i_rob, target_feat_size:target_feat_size+min(robot_feat_size, len(indices)), :] = rel_pos[:min(robot_feat_size, len(indices))]
        
    return feat_vec


def generate_data(data_size, mode='square'):
    feat_list, adj_list, label_list = [], [], []
    for _ in range(data_size):
        grid = get_reward_grid(height=HEIGHT, width=WIDTH, reward_thresh=REWARD_THRESH)
        robot_pos, adj_mat = get_initial_pose(grid, comm_range=COMM_RANGE)

        cent_act, cent_rwd = centralized_greedy_action_finder(grid, robot_pos, fov=FOV)
        action_vec = cent_act
        '''
        rand_act, rand_rwd = random_action_finder(grid, robot_pos, SAMPLE_SIZE)

        if cent_rwd > rand_rwd:
            action_vec = cent_act
        else:
            action_vec = rand_act
        '''
        if mode == 'square':
            feat_vec = get_features(grid, robot_pos, fov=FOV, step=STEP, target_feat_size=NUM_TGT_FEAT, robot_feat_size=NUM_ROB_FEAT)
        if mode == 'rect':
            feat_vec = get_rect_features(grid, robot_pos, fov=FOV, step=STEP, target_feat_size=NUM_TGT_FEAT, robot_feat_size=NUM_ROB_FEAT)
        
        feat_list.append(feat_vec)
        adj_list.append(adj_mat)

        action_one_hot = np.zeros((NUM_ROBOT, len(DIR_LIST)), dtype=np.uint8)
        action_one_hot[np.arange(NUM_ROBOT), action_vec] = 1
        label_list.append(action_one_hot)

    return [np.array(feat_list), np.array(adj_list), np.array(label_list)]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=500)
    argparser.add_argument('--save_path', type=str, default=None)
    argparser.add_argument('--mode', type=str, default='square')

    args = argparser.parse_args()

    print(f'Batch Size: {args.batch_size}')
    print(f'Data Path: {args.save_path}')
    print(f'Mode: {args.mode}')
    print('--------------- CONFIG ---------------')
    print(f'Number of robots: {NUM_ROBOT}')
    print(f'Height: {HEIGHT}')
    print(f'Width: {WIDTH}')
    print(f'Reward thresh: {REWARD_THRESH}')
    print(f'Comm Range: {COMM_RANGE}')
    print(f'FoV: {FOV}')
    print(f'Step size: {STEP}')
    print(f'#Robot in target: {NUM_TGT_FEAT}')
    print(f'#Robot in feat: {NUM_ROB_FEAT}')
    print('--------------------------------------')

    for i in tqdm(range(100)):
        pickle.dump(generate_data(args.batch_size, args.mode), open(f'{args.save_path}/data_{i+1}.pkl', 'wb'))

    print('Done!')
