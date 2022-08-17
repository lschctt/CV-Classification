import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import pow
import torch.optim.lr_scheduler as lr_scheduler
from DenseNet import DenseModel, trained_DenseModel
from GoogleNet import GoogleModel
from ResNet import ResModel, trained_ResModel


def define_optimizer(opt, model):
    optimizer = None
    if not opt.is_sep_lr:
        if opt.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        elif opt.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    elif opt.model == 'DenseNet':
        if opt.optimizer_type == 'adam':
            optimizer = torch.optim.Adam([{"params": model.features.parameters(), "lr": opt.lr1},
                                      {"params": model.linear.parameters(), "lr": opt.lr2}], weight_decay=opt.weight_decay)
        elif opt.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD([{"params": model.features.parameters(), "lr": opt.lr1},
                                          {"params": model.linear.parameters(), "lr": opt.lr2}], weight_decay=opt.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)

    return optimizer


def define_loss(opt):
    loss = None
    if opt.loss_type == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    return loss


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1
            # lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            # if (epoch - opt.niter) / float(opt.niter_decay - opt.niter) == 0.5:
            #     lr_l = 0.1
            # elif (epoch - opt.niter) / float(opt.niter_decay - opt.niter) == 0.75:
            #     lr_l = 0.1
            if epoch >= 12:
                lr_l = 0.1
            if epoch >= 18:
                lr_l = 0.01
            # if epoch >= 20:
            #     lr_l = 0.001
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_model(opt):
    model = None
    if opt.pretrained == 'none':
        if opt.model == "DenseNet":
            model = DenseModel(opt)
        elif opt.model == 'GoogleNet':
            model = GoogleModel(opt)
        elif opt.model == 'ResNet':
            model = ResModel(opt)
    else:
        # pre-trained models of densenet: densenet121, densenet169, densenet201, densenet161
        if opt.model == "DenseNet":
            pre_model = torch.hub.load('pytorch/vision:v0.10.0', opt.pretrained, pretrained=True)
            pre_model.eval()
            model = trained_DenseModel(opt, pre_model)
        # pre-trained models of resnet: resnet18, resnet34, resnet50, resnet101, resnet152
        elif opt.model == 'ResNet':
            pre_model = torch.hub.load('pytorch/vision:v0.10.0', opt.pretrained, pretrained=True)
            pre_model.eval()
            model = trained_ResModel(opt, pre_model)
        # pre-trained models of GoogleNet: googlenet
        elif opt.model == 'GoogleNet':
            model = torch.hub.load('pytorch/vision:v0.10.0', opt.pretrained, pretrained=True)
            model.eval()
    return model












