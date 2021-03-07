import numpy as np
from constants import *

def get_adjacency_matrix(robot_pos, comm_range):
    '''
    Function to get the adjacency matrix

    Parameters
    ----------
        robot_pos: Current locations for the robots
        comm_range: Maximum communication range
    
    Returns
    -------
        adj_mat: 2D array (symmetric) containing  1s and 0s to indicate if two nodes are connected
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
    adj_mat = (adj_mat > 0).astype(int)

    return adj_mat

def get_initial_pose(grid):
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
    num_robot = grid.shape[0]

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

    # Generate random location for each robot
    initial_pos = np.random.randint(low=0, high=grid.shape[0], size=(num_robot, 2))
    # Remove the rewards from the grid at robots' locations
    for pos in initial_pos:
        grid[pos[0], pos[1]] = 0

    return initial_pos

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
  
def calculate_reward(grid, robot_pos, action_list):
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
    
    # Initialize a mask of same shape as grid
    mask = np.zeros(grid.shape, dtype=int)
    
    # iterate over each robot position
    for pos in new_pos:
        # Set the values to 1 in the mask at each robot's fov
        # also make sure that the indices do not go out of grid
        mask[max(0,pos[0]-FOV):min(pos[0]+FOV+1,GRID_SIZE),
                    max(0,pos[1]-FOV):min(pos[1]+FOV+1,GRID_SIZE)] = 1
    
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
            r_pos = r_pos.clip(min=0, max=GRID_SIZE-1)

            # Find reward. It is the number of total targets in robot's FOV
            reward = np.sum(test_grid[max(0,r_pos[0]-FOV):min(r_pos[0]+FOV+1,GRID_SIZE),
                                max(0,r_pos[1]-FOV):min(r_pos[1]+FOV+1,GRID_SIZE)])
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
        test_grid[max(0,r_pos[0]-FOV):min(r_pos[0]+FOV+1,GRID_SIZE),
                    max(0,r_pos[1]-FOV):min(r_pos[1]+FOV+1,GRID_SIZE)] = 0
        
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
    # Generate random actions
    action_space = np.random.randint(low=0,high=len(DIR_DICT.keys()), size=(sample_size, NUM_ROBOT))
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

