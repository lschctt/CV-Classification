import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from define import *
from other_functions import *


# 定义Inception类
class Inception(nn.Module):
    def __init__(self, opt, inchannel):
        super(Inception, self).__init__()
        # branch1 'Pool'
        self.b1_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b1_conv = nn.Conv2d(inchannel, opt.incep_dim2, kernel_size=1)

        # branch2 'conv_1x1'
        self.b2_conv1x1 = nn.Conv2d(inchannel, opt.incep_dim2, kernel_size=1)

        # branch3 'conv_5x5'
        self.b3_conv1x1 = nn.Conv2d(inchannel, opt.incep_dim1, kernel_size=1)
        self.b3_conv5x5 = nn.Conv2d(opt.incep_dim1, opt.incep_dim2, kernel_size=5, padding=2)

        # branch4 'conv_3x3'
        self.b4_conv1x1 = nn.Conv2d(inchannel, opt.incep_dim1, kernel_size=1)
        self.b4_conv3x3_1 = torch.nn.Conv2d(opt.incep_dim1, opt.incep_dim2, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = self.b1_conv((self.b1_pool(x)))
        y2 = self.b2_conv1x1(x)
        y3 = self.b3_conv1x1(x)
        y3 = self.b3_conv5x5(y3)
        y4 = self.b4_conv1x1(x)
        y4 = self.b4_conv3x3_1(y4)
        output = [y1, y2, y3, y4]
        # torch.cat用来拼接tensor,此时数据为(b,c,w,h).维度从0开始,所以channel是维度1
        return torch.cat(output, dim=1)


class GoogleModel(torch.nn.Module):
    def __init__(self, opt):
        super(GoogleModel, self).__init__()
        input_channel = get_channel(opt)
        label_dim = get_lable_dim(opt)
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, 10, kernel_size=3), nn.MaxPool2d(2))
        # self.conv2 = nn.Sequential(nn.Conv2d(88, 20, kernel_size=5), nn.MaxPool2d(2))

        self.incep1 = Inception(opt, 10)
        self.incepList = nn.ModuleList()
        for i in range(opt.incep_num):
            incep = Inception(opt, opt.incep_dim2*4)
            self.incepList.append(incep)

        self.GAP = nn.AdaptiveAvgPool2d(5)

        self.linear = nn.Sequential(
            nn.Linear(opt.incep_dim2*4*25, label_dim),
        )

    def forward(self, x):
        batch = x.size(0)
        y_hat = F.relu((self.conv1(x)))
        y_hat = F.relu(self.incep1(y_hat))
        for incep in self.incepList:
            y_hat = incep(y_hat)
        y_hat = self.GAP(y_hat)
        y_hat = y_hat.view(batch, -1)
        y_hat = self.linear(y_hat)
        return y_hat


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class trained_GoogleModel(nn.Module):
    def __init__(self, opt, pre_trained_model):
        super(trained_GoogleModel, self).__init__()
        self.features = pre_trained_model
        self.features.classifier = Identity()

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
