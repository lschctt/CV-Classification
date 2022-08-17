import torch
import torch.nn as nn
from tqdm import tqdm
from DataLoader import *
from define import *
from options import *
from get_train_test import *


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = parse_args()
    train(opt, device)