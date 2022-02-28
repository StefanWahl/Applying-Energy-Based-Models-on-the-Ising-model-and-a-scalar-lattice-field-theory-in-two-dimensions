from turtle import forward
import torch.nn as nn
import torch
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.sparse import Embedding
import torch.nn.utils.spectral_norm as SpectralNorm

##################################################################################################################
#5 x 5
##################################################################################################################
class LinearCondition5x5_one_condition(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.convs_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels,out_channels=out_channels * 2,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.convs_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels,out_channels=out_channels * 2,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.lins_1 = nn.Sequential(
            nn.Linear(2 * out_channels,2 * out_channels),
            nn.LeakyReLU(),
            nn.Linear(2 * out_channels,1)
        )

        self.lins_2 = nn.Sequential(
            nn.Linear(2 * out_channels,2 * out_channels),
            nn.LeakyReLU(),
            nn.Linear(2 * out_channels,1)
        )

    def forward(self,X,cond_1,cond_2):
        Y = self.convs_1(X.float()).squeeze()
        Z = self.convs_2(X.float()).squeeze()

        Y = self.lins_1(Y).squeeze()
        Z = self.lins_2(Z).squeeze()

        Y = Y * cond_1 

        res = Y + Z

        return res

##################################################################################################################
#INN Subnetwork
##################################################################################################################
def fc_subnet(c_in,c_out):
    return nn.Sequential(
        nn.Linear(c_in,512),
        nn.ReLU(),
        nn.Linear(512,c_out)
    )