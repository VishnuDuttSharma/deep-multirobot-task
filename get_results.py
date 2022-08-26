import sys
import argparse

from utils.config import *

import torch
import random
import numpy as np

from agents import *
import argparse

import matplotlib
matplotlib.use('Agg')

from test_module import *
from matplotlib import pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

print('Parameters used:')
print(f'FoV: {FOV}')
print(f'Step: {STEP}')
print(f'CommR: {COMM_RANGE}')
print(f'Height: {HEIGHT}')

# parse the path of the json config file
arg_parser = argparse.ArgumentParser(description="")

arg_parser.add_argument(
    'config',
    metavar='config_json_file',
    default='None',
    help='The Configuration file in json format')

arg_parser.add_argument('--mode', type=str, default='test', help="Mode: train/test. 'test' is default")
arg_parser.add_argument('--log_time_trained', type=str, default='0')
arg_parser.add_argument('--timeid', type=str, default=None, help='Time identifier to load the model')
arg_parser.add_argument('--load_epoch', type=int, default=-1, help="Epoch number (multiple of 500) to be used for testing. If not given, latest model will be used") 

arg_parser.add_argument('--tgt_feat', type=int, default=20, help="Number of targets in feature vector. Default: 20")
arg_parser.add_argument('--rbt_feat', type=int, default=10, help="Number of neighboring robots in feature vector. Default: 20")

arg_parser.add_argument('--num_agents', type=int, default=20, help="Number of robots. Default 20")
arg_parser.add_argument('--map_w', type=int, default=20)
arg_parser.add_argument('--map_density', type=int, default=1)
arg_parser.add_argument('--map_type', type=str, default='map')

arg_parser.add_argument('--trained_num_agents', type=int, default=20)
arg_parser.add_argument('--trained_map_w', type=int, default=20)
arg_parser.add_argument('--trained_map_density', type=int, default=1)
arg_parser.add_argument('--trained_map_type', type=str, default='map')

arg_parser.add_argument('--nGraphFilterTaps', type=int, default=2, help="Number of communication filtertaps i.e. number of hops+1")
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

timeid = args.timeid


# Create the Agent and pass all the configuration to it then run it..
agent_class = globals()[config.agent]
agent = agent_class(config)

if args.load_epoch >= 0:
    filename = f'{config.save_data}/experiments/dcpOE_map20x20_rho1_{config.num_agents}Agent/K{config.nGraphFilterTaps}_HS0/{timeid}/checkpoints/checkpoint_{args.load_epoch}.pth.tar'
else:    
    filename = f'{config.save_data}/experiments/dcpOE_map20x20_rho1_{config.num_agents}Agent/K{config.nGraphFilterTaps}_HS0/{timeid}/checkpoints/checkpoint_{config.max_epoch}.pth.tar'
# filename = '/home/vishnuds/baxterB/multi_robot/gnn_log_data/dummy_model/checkpoint_500.pth.tar'
print(f'loading model from: {filename}')
checkpoint = torch.load(filename, map_location='cuda:{}'.format(agent.config.gpu_device))
agent.model.load_state_dict(checkpoint['state_dict'])
agent.model.eval()

######################################################
acc, lss = get_acc_n_loss(config, agent, agent.data_loader.train_loader)
print(f'Train Accuracy: {100*acc}%')
print(f'Train Loss: {lss}')

acc, lss = get_acc_n_loss(config, agent, agent.data_loader.validStep_loader)
print(f'Valid Accuracy: {100*acc}%')
print(f'Valid Loss: {lss}')

acc, lss = get_acc_n_loss(config, agent, agent.data_loader.test_loader)
print(f'Test Accuracy: {100*acc}%')
print(f'Test Loss: {lss}')
#######################################################

