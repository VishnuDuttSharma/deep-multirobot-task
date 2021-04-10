import sys
import argparse

from utils.config import *

import torch
import random
import numpy as np

from agents import *
import argparse

from test_module import *
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



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
config.tgt_feat = 20
config.rbt_feat = 10
config.max_epoch = 500
config.learning_rate = 0.005
config.nGraphFilterTaps = 4

timeid = args.timeid

# change line 22 and line 24 in graphs/models/coverageplanner.py

# Create the Agent and pass all the configuration to it then run it..
agent_class = globals()[config.agent]
agent = agent_class(config)

'''
print('Plotting graphs')

event_acc = EventAccumulator(f'/home/vishnuds/baxterB/multi_robot/gnn_tb_data/dcpOE_map20x20_rho1_8Agent/K4_HS0/{timeid}/')
event_acc.Reload()
train_batch_size = 64#config.batch_size
train_data_size = 60000
steps = np.array(event_acc.Scalars('iteration/loss'))[:,1].astype(int)*train_batch_size
train_loss = np.array(event_acc.Scalars('iteration/loss'))[:,2]

train_loss_avg = []
for i in range(config.max_epoch):
    # train_loss_avg.append( np.sum(train_batch_size * train_loss[(steps >= i*60000) & (steps < (i+1)*60000) ])/60000)
    train_loss_avg.append( np.mean(train_loss[(steps >= i*train_data_size) & (steps < (i+1)*train_data_size) ]))

valid_step = np.array(event_acc.Scalars('epoch/loss_validStep'))[:,1].astype(int)
valid_loss = np.array(event_acc.Scalars('epoch/loss_validStep'))[:,2]


plt.plot(np.arange(1,config.max_epoch+1), train_loss_avg, 'b')
plt.show()
plt.plot(valid_step, valid_loss, 'r')
plt.show()
'''

### /home/vishnuds/baxterB/multi_robot/gnn_log_data/experiments/dcpOE_map20x20_rho1_10Agent/K4_HS0/1617748311/

filename = f'{config.save_data}/experiments/dcpOE_map20x20_rho1_{config.num_agents}Agent/K4_HS0/{timeid}/checkpoints/checkpoint_{config.max_epoch}.pth.tar'
print(f'loading model from: {filename}')
checkpoint = torch.load(filename, map_location='cuda:{}'.format(agent.config.gpu_device))
agent.model.load_state_dict(checkpoint['state_dict'])

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

#######################################################
acc = get_stoc_acc(config, agent, agent.data_loader.train_loader)
print(f'Train Accuracy(stoc): {100*acc}%')

acc = get_stoc_acc(config, agent, agent.data_loader.validStep_loader)
print(f'Valid Accuracy(stoc): {100*acc}%')

acc = get_stoc_acc(config, agent, agent.data_loader.test_loader)
print(f'Test Accuracy(stoc): {100*acc}%')
#######################################################

#######################################################
acc = get_stoc_acc2(config, agent, agent.data_loader.train_loader)
print(f'Train Accuracy(stoc): {100*acc}%')

acc = get_stoc_acc2(config, agent, agent.data_loader.validStep_loader)
print(f'Valid Accuracy(stoc): {100*acc}%')

acc = get_stoc_acc2(config, agent, agent.data_loader.test_loader)
print(f'Test Accuracy(stoc): {100*acc}%')
########################################################
