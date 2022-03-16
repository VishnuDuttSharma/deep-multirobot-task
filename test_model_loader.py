import sys
sys.path.append('/home/vishnuds/baxterB/multi_robot/deep-multirobot-task/')

import argparse

from utils.config import *

import torch
import random
import numpy as np

from agents import *

from graphs.models.coverageplanner import CoveragePlannerNet
import time

np.random.seed(1337)
random.seed(1337)

# parse the path of the json config file
arg_parser = argparse.ArgumentParser(description="This module provides utilities for loading the pretrained models")

arg_parser.add_argument(
    'config',
    metavar='config_json_file',
    default='None',
    help='The Configuration file in json format')



arg_parser.add_argument('--mode', type=str, default='test')

arg_parser.add_argument('--tgt_feat', type=int, default=20)
arg_parser.add_argument('--rbt_feat', type=int, default=10)

arg_parser.add_argument('--num_agents', type=int, default=20)
arg_parser.add_argument('--trained_num_agents', type=int, default=20)
arg_parser.add_argument('--nGraphFilterTaps', type=int, default=2)

arg_parser.add_argument('--max_epoch', type=int, default=1500)
arg_parser.add_argument('--learning_rate', type=int, default=0.005)
arg_parser.add_argument('--log_time_trained', type=str, default='')

arg_parser.add_argument('--map_type', type=int, default=0)
arg_parser.add_argument('--best_epoch', type=int, default=None)


def get_prediction(config, feat_raw, adj_mat, device):
    
    pred_time_list = []
    pred_action_list = []

    numFeature = (config.tgt_feat + config.rbt_feat )

    feat = feat_raw
    feat_reshaped = feat[:,:,:numFeature,:].reshape(feat.shape[0], feat.shape[1], numFeature*2, order='F') 
    
    model = load_model(config)
    model.to(device);

    batch_size = 1
    for i in range(0,feat.shape[0],batch_size):
        with torch.no_grad():
            start_index = i
            end_index =  i + batch_size

            start_time = time.time()

            inputGPU = torch.FloatTensor(feat_reshaped[start_index:end_index]).to(device)
            gsoGPU =  torch.FloatTensor(adj_mat[start_index:end_index]).to(device)
            
            # model prediction
            model.addGSO(gsoGPU)
            predict = model(inputGPU)

            acts = predict.detach().cpu().numpy().argmax(axis=2)

            pred_time_list.append(time.time() - start_time)
            pred_action_list.append(acts)

    return pred_action_list, pred_time_list
    
def load_model(config):
    timeid = config.log_time_trained
    model = CoveragePlannerNet(config)

    filename = f'{config.save_data}/experiments/dcpOE_map20x20_rho1_{config.trained_num_agents}Agent/K{config.nGraphFilterTaps}_HS0/{timeid}/checkpoints/checkpoint_{config.max_epoch}.pth.tar'
    
    print(f'loading model from: {filename}')
    checkpoint = torch.load(filename, map_location='cuda:{}'.format(config.gpu_device))
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(config.device)
    model.eval()

    print('Ready for prediction\n\t pred_action_list, pred_time_list = get_prediction(config, feat_raw, adj_mat, device)')
    
    return model
    
if __name__ == "__main__":
    ####################################################################################################################
    # args = arg_parser.parse_args(['configs/coverageTask_20rob_FIX_6FOV_20STEP_10COMM_3layer_4filtertap_GNN2layer_32_128_ep1500.json'])
    args = arg_parser.parse_args()
    
    config = process_config(args)
    
    '''
    config['device'] = torch.device('cuda:0')
    config.mode = 'test'
    config.num_agents = 20
    config.tgt_feat = 20
    config.rbt_feat = 10
    config.max_epoch = 1500
    config.learning_rate = 0.005
    config.nGraphFilterTaps = 2 #4
    '''
    model = load_model(config)
    
    
    
    