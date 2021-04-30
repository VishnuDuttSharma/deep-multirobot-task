import sys

sys.path.append('/home/vishnuds/baxterB/multi_robot/deep-multirobot-task/')


import argparse

from utils.config import *
from constants import *

import torch
import random
import numpy as np

from agents import *
import argparse


from graphs.models.coverageplanner import CoveragePlannerNet
import time

import pickle
import pandas as pd

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

# parse the path of the json config file
arg_parser = argparse.ArgumentParser(description="")

arg_parser.add_argument(
    'config',
    metavar='config_json_file',
    default='None',
    help='The Configuration file in json format')



arg_parser.add_argument('--mode', type=str, default='train')
arg_parser.add_argument('--log_time_trained', type=str, default='0')
arg_parser.add_argument('--timeid', type=str, default=None)

arg_parser.add_argument('--tgt_feat', type=int, default=20)
arg_parser.add_argument('--rbt_feat', type=int, default=10)

arg_parser.add_argument('--num_agents', type=int, default=10)
arg_parser.add_argument('--map_w', type=int, default=20)
arg_parser.add_argument('--map_density', type=int, default=1)
arg_parser.add_argument('--map_type', type=str, default='map')

arg_parser.add_argument('--trained_num_agents', type=int, default=10)
arg_parser.add_argument('--trained_map_w', type=int, default=20)
arg_parser.add_argument('--trained_map_density', type=int, default=1)
arg_parser.add_argument('--trained_map_type', type=str, default='map')

arg_parser.add_argument('--nGraphFilterTaps', type=int, default=0)
arg_parser.add_argument('--hiddenFeatures', type=int, default=0)

arg_parser.add_argument('--num_testset', type=int, default=4500)
arg_parser.add_argument('--test_epoch', type=int, default=0)
arg_parser.add_argument('--lastest_epoch', action='store_true', default=False)
arg_parser.add_argument('--best_epoch', action='store_true', default=False)
arg_parser.add_argument('--con_train', action='store_true', default=False)
arg_parser.add_argument('--test_general', action='store_true', default=False)
arg_parser.add_argument('--train_TL', action='store_true', default=False)
arg_parser.add_argument('--Use_infoMode', type=int, default=0)
arg_parser.add_argument('--log_anime', action='store_true', default=False)
arg_parser.add_argument('--rate_maxstep', type=int, default=2)
arg_parser.add_argument('--commR', type=int, default=6)
np.random.seed(1337)
random.seed(1337)

args = arg_parser.parse_args()
config = process_config(args)
# print('CONFIG:')
# print(config)

config['device'] = torch.device('cuda:0')
config.mode = 'test'
config.num_agents = 20
config.tgt_feat = 20
config.rbt_feat = 10
config.max_epoch = 500
config.learning_rate = 0.005
config.nGraphFilterTaps = 5


timeid = args.timeid



model = CoveragePlannerNet(config)
filename = f'{config.save_data}/experiments/dcpOE_map20x20_rho1_{config.num_agents}Agent/K{config.nGraphFilterTaps}_HS0/{timeid}/checkpoints/checkpoint_{config.max_epoch}.pth.tar'
# filename = '/home/vishnuds/baxterB/multi_robot/gnn_log_data/dummy_model/checkpoint_500.pth.tar'
print(f'loading model from: {filename}')
checkpoint = torch.load(filename, map_location='cuda:{}'.format(config.gpu_device))
model.load_state_dict(checkpoint['state_dict'])

model = model.to(config.device)
print(model)


for inf_d in [10, 20, 30, 40 ,50]:
    graph_arr = pickle.load(open(f'./robot{config.num_agents}/inf{inf_d}/grid_data.pkl', 'rb'))
    robot_pos_arr = pickle.load(open(f'./robot{config.num_agents}/inf{inf_d}/robot_pos_data.pkl', 'rb'))
    model_data_list = pickle.load(open(f'./robot{config.num_agents}/inf{inf_d}/model_data.pkl', 'rb'))
    cg_time = pickle.load(open(f'./robot{config.num_agents}/inf{inf_d}/time_data.pkl', 'rb'))
    cg_action = pickle.load(open(f'./robot{config.num_agents}/inf{inf_d}/action_data.pkl', 'rb'))
    
    print('###')
    print(f'Inf: {inf_d}')
    print(graph_arr.shape, robot_pos_arr.shape)

    pred_time_list = []
    pred_action_list = []

    numFeature = (config.tgt_feat + config.rbt_feat )

    feat = model_data_list[0]
    feat_reshaped = feat[:,:,:numFeature,:].reshape(feat.shape[0], feat.shape[1], numFeature*2)


    batch_size = 1
    for i in range(0,feat.shape[0],batch_size):
        with torch.no_grad():
            start_index = i
            end_index =  i + batch_size

            start_time = time.time()

            inputGPU = torch.FloatTensor(feat_reshaped[start_index:end_index]).to(config.device)
            gsoGPU = torch.FloatTensor(model_data_list[1][start_index:end_index]).to(config.device)
            # gsoGPU = gsoGPU.unsqueeze(0)
            targetGPU = torch.LongTensor(model_data_list[2][start_index:end_index]).to(config.device)
            # Should not transpose if flattening the batch
            # batch_targetGPU = targetGPU.permute(1, 0, 2)
            batch_targetGPU = targetGPU
            #     agent.optimizer.zero_grad()
    
            # print('Data shapes: ', inputGPU.shape, gsoGPU.shape)
            # model
            model.addGSO(gsoGPU)
            predict = model(inputGPU)
    
            acts = predict.detach().cpu().numpy().argmax(axis=2)
    
            pred_time_list.append(time.time() - start_time)
            pred_action_list.append(acts)

            # pred_list_long.append(np.array([p.detach().cpu().numpy() for p in predict]).transpose(1,0,2))

    pred_action_list = np.concatenate(pred_action_list, axis=0)

    rand_time_list = []
    rand_action_list = []

    for i in range(feat.shape[0]):
        start_time = time.time()
        acts = np.random.randint(0,5, (1,inf_d))
        rand_time_list.append(time.time() - start_time)
        rand_action_list.append(acts)

    
    df = pd.DataFrame(index=1+np.arange(feat.shape[0]))
    df['CG_time'] = cg_time
    df['Rand_time'] = rand_time_list
    df['GNN_time'] = pred_time_list

    cg_rwd_list = []
    rnd_rwd_list = []
    pred_rwd_list = []

    for i in range(feat.shape[0]):
        grid = graph_arr[i]
        robot_pos = robot_pos_arr[i]

        rwd = calculate_reward(grid, robot_pos, cg_action[i], fov=FOV, get_mask=False)
        cg_rwd_list.append(rwd)

        rwd = calculate_reward(grid, robot_pos, rand_action_list[i][0], fov=FOV, get_mask=False)
        rnd_rwd_list.append(rwd)

        rwd = calculate_reward(grid, robot_pos, pred_action_list[i], fov=FOV, get_mask=False)
        pred_rwd_list.append(rwd)

    df['CG_reward'] = cg_rwd_list
    df['Rand_reward'] = rnd_rwd_list
    df['GNN_reward'] = pred_rwd_list

    df.to_csv(f'./summary_train_{config.num_agents}_inf_{inf_d}.csv', index=False)



