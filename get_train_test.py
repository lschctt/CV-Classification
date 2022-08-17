import torch
import torch.nn as nn
from tqdm import tqdm
from DataLoader import *
from define import *
from options import *
import matplotlib.pyplot as plt
import time


# 设定训练过程
def train(opt, device):
    model = define_model(opt)
    model = model.to(device)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    loss = nn.CrossEntropyLoss()
    data_loader = get_dataloader(opt, type='train')

    # print(model)
    print('')
    # print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    result_log = {'train': {'loss': [], 'acc': []}, 'test': {'loss': [], 'acc': []}}
    for epoch in tqdm(range(1, opt.niter + opt.niter_decay + 1)):
        num = 0
        for i, data1 in enumerate(data_loader, 0):
            (x_1, y_1) = data1
            (x_1, y_1) = (x_1.to(device), y_1.to(device))

            # forward
            y_hat = model(x_1)
            l = loss(y_hat, y_1.long())
            num += 1
            # if num % 300 == 0:
            #     print('epoch:', epoch, 'num:', num, 'Loss:', l.item())

            # backward
            optimizer.zero_grad()
            l.backward()

            # update
            optimizer.step()

        scheduler.step()
        print(scheduler.get_lr())

        acc_train = test(opt, device, type='train', model=model)
        acc_test = test(opt, device, type='test', model=model)

        result_log['train']['acc'].append(acc_train)
        result_log['test']['acc'].append(acc_test)

        print('\n[{:s}]\t\tLoss: {:s}: {:.4f}'.format('Train', 'Acc', acc_train))
        print('[{:s}]\t\tLoss: {:s}: {:.4f}\n'.format('Test', 'Acc', acc_test))


def test(opt, device, type, model):
    correct = 0
    total = 0

    data_loader = get_dataloader(opt, type=type)

    with torch.no_grad():
        for data2 in data_loader:
            (x_test, labels) = data2
            (x_test, labels) = (x_test.to(device), labels.to(device))
            y_test = model(x_test)
            _, predict = torch.max(y_test.data, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    # a.append(correct/total)
    # print('准确率：', (100 * correct / total), '%')
    return correct / total


def picture(opt, e, r_log):
    plt.subplot(2, 1, 1)
    train_loss = plt.plot(e, r_log['train']['loss'], color='red', linestyle='-.')
    test_loss = plt.plot(e, r_log['test']['loss'], color='blue', linestyle='--')

    plt.title('Acc vs. epoches(train:red)')

    t = time.strftime('%d-%H', time.localtime(time.time()))
    plt.savefig(opt.picture_save + "_model_" + opt.model + '_data_' + opt.data_name + '(' + t + ')')