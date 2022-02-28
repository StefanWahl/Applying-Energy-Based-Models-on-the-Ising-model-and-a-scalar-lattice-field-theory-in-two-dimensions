import torch.nn as nn
import torch

##################################################################################################################
#Fixed lattice size 5 x 5
##################################################################################################################

class ConvNet5x5(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels,out_channels=out_channels * 2,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2,out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2,1)
        )

    def forward(self,X):
        Y = self.convs(X.float())
        Y = self.fc(Y.squeeze()).squeeze()

        return Y

##################################################################################################################
#Fixed lattice size  20 x 20
##################################################################################################################

class ConvNet20x20(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels,out_channels=out_channels * 2,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels*2,out_channels=out_channels * 4,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels*4,out_channels=out_channels * 8,kernel_size=3,padding=1,padding_mode="circular",stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels * 8,out_channels * 8),
            nn.LeakyReLU(),
            nn.Linear(out_channels * 8,1)
        )

    def forward(self,X):
        Y = self.convs(X.float()).squeeze()
        Y = self.fc(Y.squeeze()).squeeze()

        return Y

##################################################################################################################
#Net for variable input size
##################################################################################################################

class ConvNet_multi_Version_1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="circular",stride=1,bias = False),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="circular",stride=1,bias = False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=out_channels,out_channels=out_channels * 2,kernel_size=3,padding=1,padding_mode="circular",stride=1,bias = False),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels*2,out_channels=out_channels * 2,kernel_size=3,padding=1,padding_mode="circular",stride=1,bias = False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

    def forward(self,X):
        Y = self.convs(X.float())
        Y = torch.sum(Y,(1,2,3))

        return Y