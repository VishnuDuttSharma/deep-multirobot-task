"""
Data loader for coverage task
"""


import logging
import numpy as np
import random
import torch
import pickle
from torch.utils import data
from torch.utils.data import DataLoader
from glob import glob

class GNNCoverageDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DecentralPlannerDataLoader Online Expert")
        log_info = "Loading #{} Agent DATA from Path without OnlineTF .....".format(self.config.num_agents)
        self.logger.info(log_info)
    
        if config.mode == "train":
            trainlist = [self.config.data_root+f'data_{i+1}.pkl' for i in range(1, 60)]
            train_set = GNNCoverageDataset(self.config, trainlist)
            validlist = [self.config.data_root+f'data_{i+1}.pkl' for i in range(60, 80)]
            valid_set = GNNCoverageDataset(self.config, validlist)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.valid_batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
        elif config.mode == "test":
            testlist = [self.config.data_root+f'data_{i+1}.pkl' for i in range(80, 100)]
            test_set = GNNCoverageDataset(self.config, testlist)
            self.test_loader = DataLoader(test_set, batch_size=self.config.test_batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)

        else:
            raise Exception("Please specify in the json a specified mode in mode")

    
# Data shapes reference: https://github.com/proroklab/gnn_pathplanning/blob/master/dataloader/Dataloader_dcplocal_notTF_onlineExpert.py#L147-L149
class GNNCoverageDataset(data.Dataset):
    """Coverage task dataset, generated randomly and saved"""

    def __init__(self, config, datafilelist):
        """
        Parameters
        ----------
            datafilelist: List of file of the dataset
            
        """
        numFeature = (config.tgt_feat + config.rbt_feat )
        featlist, adjlist, tgtlist = [], [], []
        for fl in datafilelist:
            feat, adj, tgt = pickle.load(open(fl, 'rb'))
            feat_reshaped = feat[:,:,:numFeature,:].reshape(feat.shape[0], feat.shape[1], numFeature*2)
            featlist.append(feat_reshaped)
            adjlist.append(adj)
            tgtlist.append(tgt)

        self.features = np.concatenate(featlist, axis=0)
        self.adj_mat  = np.concatenate(adjlist, axis=0)
        self.targets  = np.concatenate(tgtlist, axis=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.adj_mat[idx]), torch.LongTensor(self.targets[idx])

        
def generate_data(data_size):
    feat_list, adj_list, label_list = [], [], []
    for _ in range(data_size):
        grid = get_reward_grid(height=HEIGHT, width=WIDTH, reward_thresh=REWARD_THRESH)
        robot_pos, adj_mat = get_initial_pose(grid, comm_range=20)

        cent_act, cent_rwd = centralized_greedy_action_finder(grid, robot_pos, fov=FOV)
        rand_act, rand_rwd = random_action_finder(grid, robot_pos, 1000)

        if cent_rwd > rand_rwd:
            action_vec = cent_act
        else:
            action_vec = rand_act
        
        feat_vec = get_features(grid, robot_pos, fov=FOV, step=STEP, target_feat_size=NUM_TGT_FEAT, robot_feat_size=NUM_ROB_FEAT)

        feat_list.append(feat_vec)
        adj_list.append(adj_mat)
        
        action_one_hot = np.zeros((NUM_ROBOT, len(DIR_LIST)), dtype=np.uint8)
        action_one_hot[np.arange(NUM_ROBOT), action_vec] = 1
        label_list.append(action_one_hot)
    
    return [np.array(feat_list), np.array(adj_list), np.array(label_list)]
 
