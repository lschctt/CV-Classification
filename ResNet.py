import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from define import *
from other_functions import *


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.downsample != None:
            x = self.downsample(x)
        return F.relu(x + y)


def make_layers(opt, in_channels, out_channels, num, stride=1):
    downsample = None
    if (stride != 1) or (in_channels != out_channels):
        downsample = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels))
    layers = []
    layers.append(ResBlock(in_channels, out_channels, stride, downsample))
    for i in range(1, num):
        layers.append(ResBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResModel(nn.Module):
    def __init__(self, opt):
        super(ResModel, self).__init__()
        input_channel = get_channel(opt)
        # Convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, opt.res_filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(opt.res_filters[0]),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(opt.res_filters[0], opt.res_filters[1], kernel_size=3, stride=2)

        # stack of ResBlock
        self.Reslist1 = make_layers(opt=opt, in_channels=opt.res_filters[0], out_channels=opt.res_filters[0],
                                    num=opt.res_n[0], stride=1)
        self.Reslist2 = make_layers(opt=opt, in_channels=opt.res_filters[0], out_channels=opt.res_filters[1],
                                    num=opt.res_n[1], stride=2)
        self.Reslist3 = make_layers(opt=opt, in_channels=opt.res_filters[1], out_channels=opt.res_filters[2],
                                    num=opt.res_n[2], stride=2)

        # pooling
        self.MP = nn.MaxPool2d(2)
        self.AP = nn.AdaptiveAvgPool2d(1)

        #linear
        label_dim = get_lable_dim(opt)
        self.linear = nn.Linear(opt.res_filters[2], label_dim)

    def forward(self, x):
        y_hat = self.conv1(x)
        y_hat = self.Reslist1(y_hat)
        y_hat = self.Reslist2(y_hat)
        y_hat = self.Reslist3(y_hat)
        y_hat = self.AP(y_hat)
        y_hat = y_hat.view(y_hat.size(0), -1)
        y_hat = self.linear(y_hat)
        return y_hat


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class trained_ResModel(nn.Module):
    def __init__(self, opt, pre_trained_model):
        super(trained_ResModel, self).__init__()
        self.features = pre_trained_model
        self.features.fc = Identity()

        label_dim = get_lable_dim(opt)
        pretrain_dim = get_pretrain_dim(opt)
        self.linear = nn.Sequential(
            nn.Linear(pretrain_dim, label_dim),
            nn.BatchNorm1d(label_dim),
            # nn.Dropout(0.25)
        )

    def forward(self, x):
        y_hat = self.features(x)
        # print(y_hat.shape)
        y_hat = self.linear(y_hat)
        return y_hat





