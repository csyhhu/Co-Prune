"""
Test the accuracy of FP network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse

from models.CIFARNet import CIFARNet
from torch.autograd import Variable
from utils.train import progress_bar
from utils.dataset import get_dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', '-m', type=str, default='CIFARNet', help='Model Arch')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR10', help='Dataset')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model_name = args.model
dataset_name = args.dataset
pretrain_path = './Results/%s-%s/%s-%s-retrain.pth' %(model_name, dataset_name, model_name, dataset_name)

# Data
print('==> Preparing data..')
testloader = get_dataloader(dataset_name, 'test', 100)
if model_name == 'CIFARNet':
    if dataset_name in ['CIFAR10', 'STL10']:
        net = CIFARNet(num_classes=10)
    elif dataset_name in ['CIFAR9', 'STL9']:
        net = CIFARNet(num_classes=9)
    else:
        raise ('%s in %s have not been finished' % (model_name, dataset_name))
else:
    raise NotImplementedError

net.load_state_dict(torch.load(pretrain_path))

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def test(net, testloader):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = net(inputs)

        loss = nn.CrossEntropyLoss()(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))

test(net, testloader)