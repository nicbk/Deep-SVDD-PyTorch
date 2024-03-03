import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


# Network copied from:
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb
class CelebA_Net(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        modules = []
        hidden_dims = [32, 64, 128]

        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*20*20, self.rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CelebA_Net_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        modules_enc = []
        hidden_dims_enc = [32, 64, 128]

        in_channels_enc = 3
        for h_dim in hidden_dims_enc:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels_enc = h_dim

        self.encoder = nn.Sequential(*modules_enc)
        self.fc_enc = nn.Linear(hidden_dims_enc[-1]*20*20, self.rep_dim)

        modules_dec = []
        hidden_dims_dec = [64, 32, 3]
        self.fc_dec = nn.Linear(self.rep_dim, hidden_dims_dec[-1]*20*20)

        in_channels_dec = 128
        self.in_channels_dec = in_channels_dec
        for (i, h_dim) in enumerate(hidden_dims_dec):
            modules_dec.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                )
            )
            in_channels_dec = h_dim

        self.decoder = nn.Sequential(*modules_dec)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_enc(x)

        x = self.fc_dec(x)
        x = x.view(x.size(0), self.in_channels_dec*20*20, 20, 20)
        x = self.decoder()
        x = torch.sigmoid(x)
        return x
