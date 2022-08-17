def get_channel(opt):
    input_channel = None
    if opt.input_channel != 0:
        input_channel = opt.input_size
    else:
        if opt.data_name == 'MNIST':
            input_channel = 1
        elif opt.data_name == 'CIFAR10':
            input_channel = 3
        elif opt.data_name == 'CIFAR100':
            input_channel = 3

    return input_channel


def get_lable_dim(opt):
    label_dim = None
    if opt.label_dim != 0:
        label_dim = opt.label_dim
    else:
        if opt.data_name == 'MNIST':
            label_dim = 10
        elif opt.data_name == 'CIFAR10':
            label_dim = 10
        elif opt.data_name == 'CIFAR100':
            label_dim = 100

    return label_dim


def get_pretrain_dim(opt):
    pretrain_dim = None
    if opt.pretrained == 'densenet121':
        pretrain_dim = 1024
    elif opt.pretrained == 'densenet169':
        pretrain_dim = 1664
    elif opt.pretrained == 'resnet18':
        pretrain_dim = 512
    elif opt.pretrained == 'resnet50':
        pretrain_dim = 2048
    return pretrain_dim

