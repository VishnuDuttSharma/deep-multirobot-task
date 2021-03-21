import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.weights_initializer import weights_init
import numpy as np
import utils.graphUtils.graphML as gml
import utils.graphUtils.graphTools
from torchsummaryX import summary

class CoveragePlannerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.S = None
        self.numAgents = self.config.num_agents
        # inW = self.config.map_w
        # inH = self.config.map_h
        numFeatures = (self.tgt_feat + self.rbt_feat)*2
        numAction = 5
        # ------------------ DCP v1.5  -  no CNN- less feature
        dimCompressMLP = 2
        numCompressFeatures = [2 ** 4, 2 ** 3]
        # # 1 layer origin
        dimNodeSignals = [2 ** 4]

        
        nGraphFilterTaps = [self.config.nGraphFilterTaps]
        # --- actionMLP
        dimActionMLP = 1
        numActionFeatures = [numAction]



        #####################################################################
        #                                                                   #
        #                MLP-feature compression                            #
        #                                                                   #
        #####################################################################

        numCompressFeatures = [numFeatures] + numCompressFeatures

        compressmlp = []
        for l in range(dimCompressMLP):
            compressmlp.append(
                nn.Linear(in_features=numCompressFeatures[l], out_features=numCompressFeatures[l + 1], bias=True))
            compressmlp.append(nn.ReLU(inplace=True))

        self.compressMLP = nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [numCompressFeatures[-1]] + dimNodeSignals  # Features
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(gml.GraphFilterBatch(self.F[l], self.F[l + 1], self.K[l], self.E, self.bias))
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            gfl.append(nn.ReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        numActionFeatures = [self.F[-1]] + numActionFeatures
        actionsfc = []
        for l in range(dimActionMLP):
            if l < (dimActionMLP - 1):
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))
                actionsfc.append(nn.ReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))

        self.actionsMLP = nn.Sequential(*actionsfc)
        self.apply(weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers)


    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

    def forward(self, inputTensor):

        B = inputTensor.shape[0] # batch size

        # B x G x N
        extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(self.config.device)
        for id_agent in range(self.numAgents):
            input_currentAgent = inputTensor[:, id_agent]
            # featureMap = self.ConvLayers(input_currentAgent)
            # featureMapFlatten = featureMap.view(featureMap.size(0), -1)
            # extractFeatureMap[:, :, id_agent] = featureMapFlatten
            # compressfeature = self.compressMLP(featureMapFlatten)
            compressfeature = self.compressMLP(input_currentAgent)
            extractFeatureMap[:, :, id_agent] = compressfeature # B x F x N

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)

        action_predict = []
        for id_agent in range(self.numAgents):
            # DCP_nonGCN
            # sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
            # DCP
            # torch.index_select(sharedFeature_currentAgent, 3, id_agent)
            sharedFeature_currentAgent = sharedFeature[:, :, id_agent]
            # print("sharedFeature_currentAgent.requires_grad: {}\n".format(sharedFeature_currentAgent.requires_grad))
            # print("sharedFeature_currentAgent.grad_fn: {}\n".format(sharedFeature_currentAgent.grad_fn))

            sharedFeatureFlatten = sharedFeature_currentAgent.view(sharedFeature_currentAgent.size(0), -1)
            action_currentAgents = self.actionsMLP(sharedFeatureFlatten) # 1 x 5
            action_predict.append(action_currentAgents) # N x 5


        return action_predict
