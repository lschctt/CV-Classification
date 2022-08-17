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
        tran = transforms.Compose([transforms.ToTensor()])
    else:
        tran = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    is_train = True if type == 'train' else False
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