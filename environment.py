import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from constants import *

# NUM_ROBOT = 2 #number of robots
# REWARD_THRESH = 0.70 # mimimum value of a reward
# PATH_LEN = 2 # Path length/Time horizon
# HEIGHT = 16 # Heigh of the grid
# WIDTH = 16 # Width of the grid


# Dictionary to save directions
# DIR_DICT = {
#     0: np.array([-1,0]), #up
#     1: np.array([ 1,0]), # down
#     2: np.array([0,-1]), # left
#     3: np.array([0, 1]) # right
# }

# # List containing numpy array orresponding to the directions
# DIR_LIST = np.array(list(DIR_DICT.values()))


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

    # Apply mask over the reward grid
    return reward_grid * mask


def is_covered(point, path):
    """
    Function to check if a point/location is already on a path

    Parameters
    ----------
        point: Point/Location to be queries
        path: Path

    Return
    ------
        True is point lies on the path, else False
    """
    # Iterate through every point/location on path
    for i in range(len(path)):
        if (path[i] == point).all(): # Check if points are same
            return True
    return False

def get_children(cur_path, limits=(HEIGHT, WIDTH)):
    """
    Function to get leaves of a tree node

    Parameters
    ----------
        cur_path: Current Path; path explored so far for a node
        limits: width and height of the grid

    Returns
    -------
        child_paths: Path from node to the children/leaves
    """
    # Create empty array to hold path through children
    child_paths = []

    # Search in each direction
    for i in range(4):
        # Move from the current location (last point in the path) in a direction
        temp_pos = cur_path[-1] + DIR_DICT[i]

        # Check if the new point/location is already on the path
        if is_covered(temp_pos,cur_path):
            continue # skip
        # Check if the new point/location is withoin bpunds
        elif ((temp_pos < 0).any() or (temp_pos[0] >= limits[0]) or (temp_pos[1] >= limits[1])):
            continue # skip
        # Add the child node/leaf to the path
        else:
            child_paths.append(cur_path + [temp_pos])

    return child_paths


def recursive_exploration(parent_pos, DEPTH):
    """
    Function to recursively explore node (tree) to generate candidate paths

    Parameters
    ----------
        parent_pos: Locarion of the parent node
        DEPTH: Current depth level

    Returns
    -------
        child_paths: All candidate paths from the parent pos
    """
    # If depth is 1 return the current node
    if DEPTH == 1:
        return parent_pos

    # Create empty array to hold path through children
    child_paths = []

    # Generate children/leave for the current point/node
    children = get_children(parent_pos)

    ## For debugging
    # print(DEPTH, len(children), children)

    for child in children:
        ## For debugging
        # print(child)

        # Recursively explore the children
        child_paths.append(recursive_exploration(child, DEPTH-1))

        ## For debugging
        # print(child_paths)
        # print('####')

    return child_paths

def get_best_path(path_combs, grid):
    """
    Function ot get the best path from all path combinations

    Parameters
    ----------
        path_combs: Combination of paths (list containing two tuples: one for each robot paths)
        grid: Reward Grid

    Returns
    -------
        Best path
    """
    # Empty array to contain reward collected over each path combinations
    cost_list = []

    # Iterate over each combination
    for path in path_combs:
        ## Debugging
        # print('Raw Path')
        # print(np.array(path).reshape(-1,2))

        # convert the path combination into a 2D array of unique points on both baths
        points = np.unique(np.array(path).reshape(-1,2), axis=0)
        # Find the total reward at each point on the grid
        cost_list.append( np.sum(grid[points[:,0], points[:,1]]) )

        ## Debugging
        # print('Path')
        # print(path)
        # print('Points')
        # print(type(points))
        # print(points)
        # print('Cost')
        # print(cost_list[-1])
        # print('#'*10)

    # Return the best path using argmax
    return path_combs[np.argmax(cost_list)]

def run_bfs(intial_pos, grid, path_len=PATH_LEN):
    """
    Function to run BFS

    Parameters
    ----------
        intial_pos: Initial location of each robot
        grid: Reward grid

    Returns
    -------
        best_comb: best path for the given reward grid and initial locations
    """
    # Empty array to hold all paths
    path_list_big = []

    # Iterate over each robot
    for i in range(NUM_ROBOT):
        # Empty array to fold all candidate paths for a robots
        path_list = []
        # Find all candidate paths for i^{th} robot
        temp_path = recursive_exploration([intial_pos[i]], path_len)

        # Add all candidate paths into list for all paths
        for path in temp_path:
            if len(path) == 2:
                path_list.extend([path])
            else:
                path_list.extend(path)

        path_list_big.append(path_list)

    # Get all paths combinations
    path_combinations =  list(product(*path_list_big))

    # Get the bets path combination
    best_comb = get_best_path(path_combinations, grid)

    return best_comb

def to_feature(gt_path, grid):
    """
    Function to convert the ground truth paths (BFS outputs) to features and action

    Parameters
    ----------
        gt_path: Ground truth paths (BFS outputs). It is a list containing NUM_ROBTOS paths, each of length PATH_LEN
        grid: Reward grid

    Returns
    -------
        feat_list: list containing features (rewards in four directions) for each robot
        act_list : list containing actions for each robot
    """
    # Empty arrays to save features and actions
    feat_list = []
    act_list = []

    # Iterate over each robot
    for i in range(NUM_ROBOT):
        # Current position = First location on the path = Inital location of i^{th} robot
        cur_pos = gt_path[i][0]

        # Features in each direction
        hf1 = np.array([np.sum(grid[0:cur_pos[0]-1+1, :]), #up
                        np.sum(grid[cur_pos[0]+1:, :]), # down
                        np.sum(grid[:, 0:cur_pos[1]-1+1]), # left
                        np.sum(grid[:, cur_pos[1]+1:])]) #right

        # Get different in the first and second location to get action
        diff_arr = gt_path[i][1] - cur_pos # action at one time-step

        ## Debugging
        # print(gt_path[i][1], cur_pos, diff_arr)
        # print(np.where((DIR_LIST == diff_arr).all(axis=1)))
        # print(diff_arr)

        # Convert the location differnce into action (0,1,2,3) by matching with the direction list
        action = np.where((DIR_LIST == diff_arr).all(axis=1))[0][0] # action at one time-step

        # Add features and action to losts
        feat_list.append(hf1)
        act_list.append(action)

    return feat_list, act_list

def to_fov_feature(gt_path, grid, fov, num_robot=NUM_ROBOT):
    """
    Function to convert the ground truth paths (BFS outputs) to features and action using fov

    Parameters
    ----------
        gt_path: Ground truth paths (BFS outputs). It is a list containing NUM_ROBTOS paths, each of length PATH_LEN
        grid: Reward grid
        fov: Field-of-View ('fov' number of pixels on each side. fov=3 means grid of size 7x7)

    Returns
    -------
        feat_list: list containing features (rewards in four directions) for each robot
        act_list : list containing actions for each robot
    """
    # Empty arrays to save features and actions
    feat_list = []
    act_list = []

    # Iterate over each robot to find fov based grid
    for i in range(num_robot):
        # Current position = First location on the path = Inital location of i^{th} robot
        cur_pos = gt_path[i][0]

        # Create a indicator grid
        temp_grid = np.zeros(grid.shape, dtype=grid.dtype)
        # Set 1s in fov around the robot
        temp_grid[ max(0, cur_pos[0]-fov) :  min(cur_pos[0] + fov + 1, grid.shape[0]), max(0, cur_pos[1]-fov) :  min(cur_pos[1] + fov + 1, grid.shape[1])]  = 1

    # update grid to keep only those values which fall in the FOV
    grid = temp_grid * grid

    # Iterate over each robot
    for i in range(num_robot):
        # Current position = First location on the path = Inital location of i^{th} robot
        cur_pos = gt_path[i][0]

        # Features in each direction
        hf1 = np.array([np.sum(grid[0:cur_pos[0]-1+1, :]), #up
                        np.sum(grid[cur_pos[0]+1:, :]), # down
                        np.sum(grid[:, 0:cur_pos[1]-1+1]), # left
                        np.sum(grid[:, cur_pos[1]+1:])]) #right

        # Get different in the first and second location to get action
        diff_arr = gt_path[i][1] - cur_pos # action at one time-step

        ## Debugging
        # print(gt_path[i][1], cur_pos, diff_arr)
        # print(np.where((DIR_LIST == diff_arr).all(axis=1)))
        # print(diff_arr)

        # Convert the location differnce into action (0,1,2,3) by matching with the direction list
        action = np.where((DIR_LIST == diff_arr).all(axis=1))[0][0] # action at one time-step

        # Add features and action to losts
        feat_list.append(hf1)
        act_list.append(action)

    return feat_list, act_list

def to_gnn_feature(gt_path, grid, fov, num_robot=NUM_ROBOT, feat_len=NUM_GNN_FEAT, local=True):
    """
    Function to convert the ground truth paths (BFS outputs) to features and action using fov

    Parameters
    ----------
        gt_path: Ground truth paths (BFS outputs). It is a list containing NUM_ROBTOS paths, each of length PATH_LEN
        grid: Reward grid
        fov: Field-of-View ('fov' number of pixels on each side. fov=3 means grid of size 7x7): NOT USED
        num_robot: number of robots
        feat_len: length of features/closest reward locations to be saved
        local: If True, location w.r.t. robot location will be saved. If false, global location will be saved
    Returns
    -------
        feat_list: list containing features (rewards locations in four directions) for each robot
        act_list : list containing actions for each robot
    """
    # Empty arrays to save features and actions
    feat_list = np.ones(shape=(num_robot, 4, feat_len, 2))*-1
    act_list = []

    # Iterate over each robot to find fov based grid
    for i in range(num_robot):
        # Current position = First location on the path = Inital location of i^{th} robot
        cur_pos = gt_path[i][0]

        ## Get masked version of grid in each direction
        
        # UP: all cells below current row (inclusive) = 0
        h_mask_up = np.zeros(grid.shape)
        h_mask_up[max(0,cur_pos[0]-STEP-fov):min(cur_pos[0]+fov+1, grid.shape[0]), max(0,cur_pos[1]-fov):min(cur_pos[1]+fov+1, grid.shape[1])] = 1
        h_mask_up = grid*h_mask_up         

        # DOWN: all cells above current row (inclusive) = 0
        h_mask_down = np.zeros(grid.shape)
        h_mask_down[max(0,cur_pos[0]-fov):min(cur_pos[0]+STEP+fov+1, grid.shape[0]), max(0,cur_pos[1]-fov):min(cur_pos[1]+fov+1, grid.shape[1])] = 1
        h_mask_down = grid*h_mask_down

        # LEFT: all cells to the right of current column (inclusive) = 0
        h_mask_left = np.zeros(grid.shape)
        h_mask_left[max(0,cur_pos[0]-fov):min(cur_pos[0]+fov+1, grid.shape[0]), max(0,cur_pos[1]-STEP-fov):min(cur_pos[1]+fov+1, grid.shape[1])] = 1
        h_mask_left = grid*h_mask_left

        # RIGHT: all cells to the left of the current cilumn (inclusive) = 0
        h_mask_right = np.zeros(grid.shape)
        h_mask_right[max(0,cur_pos[0]-fov):min(cur_pos[0]+fov+1, grid.shape[0]), max(0,cur_pos[1]-fov):min(cur_pos[1]+STEP+fov+1, grid.shape[1])] = 1
        h_mask_right = grid*h_mask_right        

        # Iterate over each mask
        for idx, h_mask in enumerate([h_mask_up, h_mask_down, h_mask_left, h_mask_right]):
            # Get locations of all the non-zero locations (reward locations)
            r_rows, r_cols = np.where(h_mask > 0)
            # Convert them to 2d coordinates
            coords = np.stack([r_rows, r_cols], axis=1) ## num_rewards, 2
            # Sort them by distance from current locaition (closest first)
            argorder = np.argsort(np.linalg.norm(coords - cur_pos, axis=1))
            
            # if local==true, convert corrdinates to relative locations
            if local:
                coords = coords - cur_pos
            # Get coordinates in the sorted order and scale integer locations into float (width, height->1,1)
            feats = coords[argorder] / np.array(grid.shape)

            # Add them to features, use `feat_len` locations at max
            feat_list[i, idx,:min(len(feats), feat_len), :] = feats[:min(len(feats), feat_len)]

        # Get different in the first and second location to get action
        diff_arr = gt_path[i][1] - cur_pos # action at one time-step
        # Convert the location differnce into action (0,1,2,3) by matching with the direction list
        action = np.where((DIR_LIST == diff_arr).all(axis=1))[0][0] # action at one time-step

        # Add features and action to losts
        act_list.append(action)
    
    return feat_list, act_list

