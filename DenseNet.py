import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from define import *
from other_functions import *


# DenseNet
# element of DenseBlock
class H(nn.Module):
    def __init__(self, inchannel, k):
        super(H, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, 4 * k, kernel_size=1)
        self.conv2 = nn.Conv2d(4 * k, k, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(inchannel)
        self.BN2 = nn.BatchNorm2d(4 * k)

    def forward(self, x):
        y = self.conv1(F.relu(self.BN1(x)))
        y = self.conv2(F.relu(self.BN2(y)))
        return y


class DenseBlock(nn.Module):
    def __init__(self, L, k0, k):
        super(DenseBlock, self).__init__()
        self.begin = False
        self.HList = nn.ModuleList()
        for i in range(L):
            Hi = H(k0 + k * i, k)
            self.HList.append(Hi)

    def forward(self, x):
        for layer in self.HList:
            y = layer(x)
            x = torch.cat((y, x), dim=1)
        return x


class Transition(nn.Module):
    def __init__(self, m):
        super(Transition, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(m, m // 2, kernel_size=1),
            nn.BatchNorm2d(m // 2),
            nn.ReLU(True),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        y = self.conv1(x)
        return y


class DenseModel(nn.Module):
    def __init__(self, opt):
        super(DenseModel, self).__init__()
        # Convolution layer
        input_channel = get_channel(opt)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, opt.c0, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(opt.c0),
            nn.ReLU(True),
        )
        # Pooling layer
        self.GAP = nn.AdaptiveAvgPool2d(1)

        # Densely Block
        len_Dense = len(opt.K)
        k0 = []   # use to get the initial channel of DenseBlock
        T = []    # use to get the initial channel of Transition
        k0.append(opt.c0)
        T.append(opt.K[0] * opt.L[0] + opt.c0)
        for i in range(1, len_Dense):
            k0.append(int(T[i-1]/2))
            T.append(int(opt.K[i] * opt.L[i] + k0[i]))

        self.DTList = nn.ModuleList()
        for i in range(len_Dense):
            DB = DenseBlock(L=opt.L[i], k0=k0[i], k=opt.K[i])
            self.DTList.append(DB)
            if i != len_Dense-1:
                tran = Transition(T[i])
                self.DTList.append(tran)
        self.DTList.append(nn.BatchNorm2d(T[len_Dense-1]))
        self.DTList.append(nn.ReLU(True))

        # linear
        label_dim = get_lable_dim(opt)
        input_dim = T[len_Dense-1]
        self.decoder = nn.Sequential(nn.Linear(input_dim, label_dim), nn.AlphaDropout(p=0.25))

    def forward(self, x):
        batch = x.size(0)  # get the batch_size of x
        y = self.conv1(x)
        for layer in self.DTList:
            y = layer(y)
        y = self.GAP(y)
        y = y.view(batch, -1)
        y = self.decoder(y)
        return y


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class trained_DenseModel(nn.Module):
    def __init__(self, opt, pre_trained_model):
        super(trained_DenseModel, self).__init__()
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


