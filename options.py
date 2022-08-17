import argparse
import os

import torch


def print_options(parser, opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


# def parse_gpuids(opt):
#     # set gpu ids
#     str_ids = opt.gpu_ids.split(',')
#     opt.gpu_ids = []
#     for str_id in str_ids:
#         id = int(str_id)
#         if id >= 0:
#             opt.gpu_ids.append(id)
#     if len(opt.gpu_ids) > 0:
#         torch.cuda.set_device(opt.gpu_ids[0])
#
#     return opt

def parse_args():
    parser = argparse.ArgumentParser()
    # arguments for all
    parser.add_argument('--dataroot', default='./data', type=str, help="datasets")
    parser.add_argument('--model', default='GoogleNet', type=str, help="model name")
    parser.add_argument('--loss_type', default='CrossEntropyLoss', type=str, help="type of loss")
    parser.add_argument('--data_name', default='MNIST', type=str)
    # pre_trainde models: densenet121, densenet169, resnet18, resnet50, googlenet
    parser.add_argument('--pretrained', default='none', type=str, help="whether use pre-trained model")
    parser.add_argument('--is_picture', default=0, type=int, help="whether save pictures")
    parser.add_argument('--label_dim', default=0, type=int)
    parser.add_argument('--input_channel', default=0, type=int)
    parser.add_argument('--drop_rate', default=0.25, type=float)
    parser.add_argument('--optimizer_type', default='sgd', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_policy', default='linear', type=str, help='decay method of lr')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--reg_type', default='all', type=str, help="regularization type")
    parser.add_argument('--lr_decay', default='linear', type=str, help='decay way of lr')
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=25, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--is_sep_lr', default=0, type=int, help='if use two lr to train encoder and decoder')
    parser.add_argument('--lr1', default=3e-4, type=float, help='the lr to train encoder')
    parser.add_argument('--lr2', default=0.005, type=float, help='the lr to train decoder')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use')
    parser.add_argument('--weight_decay', default=4e-4, type=float, help='Used for Adam. L2 Regularization on weights')
    parser.add_argument('--momentum', default=0.9, type=float, help='Used for sgd')

    # arguments for densenet
    parser.add_argument('--L', default=[32, 32, 32], type=int, help='number of DenseBlock', nargs='+')
    parser.add_argument('--K', default=[12, 12, 12], type=int, help='number of DenseBlock', nargs='+')
    parser.add_argument('--c0', default=16, type=int, help='the output channel of Conv')

    # arguments for googlenet
    parser.add_argument('--incep_num', default=3, type=int, help='the number of Inception')
    parser.add_argument('--incep_dim1', default=16, type=int, help='the output dim of each part in Inception')
    parser.add_argument('--incep_dim2', default=24, type=int, help='the output dim of each part in Inception')

    # argument for resnet
    parser.add_argument('--res_filters', default=[16, 32, 64], type=int, help='input channel of ResBlock', nargs='+')
    parser.add_argument('--res_n', default=[3, 3, 3], type=int, help='number of ResBlock', nargs='+')

    args = parser.parse_args()
    print_options(parser, args)
    # args = parse_gpuids(args)

    return args








