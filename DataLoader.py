import options
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def get_root(opt):
    return opt.dataroot + '/' + opt.data_name + '/'


def get_dataset(opt, type):
    dataset = None
    if opt.pretrained == 'none':
        tran_train = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                   # 图像一半的概率翻转，一半的概率不翻转
                                   transforms.RandomHorizontalFlip()])
        tran_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        tran_train = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
        tran_test = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    if type == 'train':
        is_train = True
        tran = tran_train
    else:
        is_train = False
        tran = tran_test

    root = get_root(opt)

    if opt.data_name == 'MNIST':
        dataset = datasets.MNIST(root=root, train=is_train, download=True, transform=tran)
    elif opt.data_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root=root, train=is_train, download=True, transform=tran)
    elif opt.data_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root=root, train=is_train, download=True, transform=tran)

    return dataset


def get_dataloader(opt, type):
    isshuffle = True if type == 'train' else False
    ds = get_dataset(opt, type)
    dl = DataLoader(dataset=ds, shuffle=isshuffle, batch_size=opt.batch_size)
    return dl