import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class COMPAS_Net(BaseNet):
    def __init__(self, num_tags):
        super().__init__()
        self.rep_dim = 16

        self.e_hidden1 = nn.Linear(num_tags, 64)
        self.e_hidden2 = nn.Linear(64, 32)
        self.e_hidden3 = nn.Linear(32, self.rep_dim)

    def forward(self, x):
        x = F.leaky_relu(self.e_hidden1(x))
        x = F.leaky_relu(self.e_hidden2(x))
        x = F.leaky_relu(self.e_hidden3(x))

        return x

class COMPAS_Net_Autoencoder(BaseNet):

    def __init__(self, num_tags):
        super().__init__()
        self.rep_dim = 16

        self.e_hidden1 = nn.Linear(num_tags, 64)
        self.e_hidden2 = nn.Linear(64, 32)
        self.e_hidden3 = nn.Linear(32, self.rep_dim)

        self.d_hidden1 = nn.Linear(self.rep_dim, 32)
        self.d_hidden2 = nn.Linear(32, 64)
        self.d_hidden3 = nn.Linear(64, num_tags)

    def forward(self, x):
        x = F.leaky_relu(self.e_hidden1(x))
        x = F.leaky_relu(self.e_hidden2(x))
        x = F.leaky_relu(self.e_hidden3(x))

        x = F.leaky_relu(self.d_hidden1(x))
        x = F.leaky_relu(self.d_hidden2(x))
        x = torch.sigmoid(self.d_hidden3(x))

        return x
