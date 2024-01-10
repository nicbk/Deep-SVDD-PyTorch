import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class COMPAS_Net(BaseNet):
    def __init__(self, num_tags):
        super().__init__()

        self.e_hidden1 = nn.Linear(num_tags, 32)
        self.e_hidden2 = nn.Linear(32, 16)

    def forward(self, x):
        x = F.leaky_relu(self.e_hidden1(x))
        x = F.leaky_relu(self.e_hidden2(x))

        return x

class COMPAS_Net_Autoencoder(BaseNet):

    def __init__(self, num_tags):
        super().__init__()

        self.e_hidden1 = nn.Linear(num_tags, 32)
        self.e_hidden2 = nn.Linear(32, 16)

        self.d_hidden1 = nn.Linear(16, 32)
        self.d_hidden2 = nn.Linear(32, num_tags)

    def forward(self, x):
        x = F.leaky_relu(self.e_hidden1(x))
        x = F.leaky_relu(self.e_hidden2(x))

        x = F.leaky_relu(self.d_hidden1(x))
        x = torch.sigmoid(self.d_hidden2(x))

        return x
