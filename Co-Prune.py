"""
A relatively finalized code for co-prune with adaptive decrease alpha (alpha represents how much information is transfer).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.CIFARNet import SparseCIFARNet
from utils.dataset import get_dataloader
from utils.train import progress_bar, accuracy
from utils.sparse import mask_test
from utils.recorder import Recorder

import itertools
import pickle
import shutil
import numpy as np
import time

import argparse
parser = argparse.ArgumentParser(description='Co-Prune')
parser.add_argument('--model', '-m', type=str, default='CIFARNet', help='Model')
parser.add_argument('--source', '-s', type=str, default='CIFAR9', help='Source Domain')
parser.add_argument('--target', '-t', type=str, default='STL9', help='Target Domain')
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--n_epoch', '-n', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--alpha','-a', nargs='+', help='Variation of alpha', required=True)
args = parser.parse_args()

# ---------------------------------------------------------------
use_cuda = torch.cuda.is_available()
model_name = args.model
source_dataset_name = args.source
target_dataset_name = args.target
save_root = './Results/%s-%s-%s' %(model_name, source_dataset_name, target_dataset_name)
source_pretrain_path = '%s/%s-%s-pretrain.pth' \
                       %(save_root, model_name, source_dataset_name)
target_pretrain_path = '%s/%s-%s-pretrain.pth' \
                       %(save_root, model_name, source_dataset_name)
n_epoch = args.n_epoch
exp_spec = args.exp_spec
alpha_list = args.alpha
# ----------------------------------------------------------------

##############
# Specify CR #
##############
from CR_setting import get_sparse_param_analysis
from CR_setting import CR_ratio_1_35 as CR_ratio
CR_list = [CR_ratio] * len(alpha_list)
# print(alpha_list)
for idx, alpha in enumerate(alpha_list):
    alpha_list[idx] = float(alpha)

#########################
# Load Pretrained Model #
#########################
if source_dataset_name in ['CIFAR10']:
    num_classes = 10
elif source_dataset_name in ['CIFAR9']:
    num_classes = 9
else:
    raise NotImplementedError

source_net = SparseCIFARNet(num_classes=num_classes)
target_net = SparseCIFARNet(num_classes=num_classes)

source_pretrain_param = torch.load(source_pretrain_path)
target_pretrain_param = torch.load(target_pretrain_path)

source_net.load_state_dict(source_pretrain_param)
target_net.load_state_dict(target_pretrain_param)

overall_CR = get_sparse_param_analysis(target_net, CR_ratio)

if use_cuda:
    source_net.cuda()
    target_net.cuda()

################
# Load Dataset #
################
source_loader = get_dataloader(source_dataset_name, 'train', 128)
target_loader = get_dataloader(target_dataset_name, 'train', 128)
source_test_loader = get_dataloader(source_dataset_name, 'test', 100)
target_test_loader = get_dataloader(target_dataset_name, 'test', 100)

#####################
# Initial mask dict #
#####################
source_mask_dict = dict()
target_mask_dict = dict()

#####################
# Initial Recording #
#####################
source_summary_path = '%s/runs-%s/CR%.2f' %(save_root, source_dataset_name, 100 * overall_CR)
target_summary_path = '%s/runs-%s/CR%.2f' %(save_root, target_dataset_name, 100 * overall_CR)

for SummaryPath in [source_summary_path, target_summary_path]:

    if args.exp_spec is not '':
        SummaryPath += ('-' + args.exp_spec)

    if os.path.exists(SummaryPath):
        print('Record exist, remove')
        input()
        shutil.rmtree(SummaryPath)
        os.makedirs(SummaryPath)
    else:
        os.makedirs(SummaryPath)

source_recorder = Recorder(SummaryPath=source_summary_path, dataset_name=source_dataset_name)
target_recorder = Recorder(SummaryPath=target_summary_path, dataset_name=source_dataset_name)
alpha_change_point_file = open('%s/alpha_change_point.txt' %(target_summary_path), 'w+')

##################
# Begin Training #
##################
best_acc_list = [] # Best test acc in each training period under various alpha
niter = 0
for ite, CR_ratio in enumerate(CR_list):

    alpha = alpha_list[ite]

    print('Adaptive iteration: %d, alpha: %.3f' % (ite, alpha))
    print('Current CR: %s' % CR_ratio)
    print('niter: %d' %niter)

    if ite != 0:
        print('Loading best model from previous alpha')
        source_net.load_state_dict(torch.load(
            '%s/checkpoint/%s-temp.pth' %(save_root, source_dataset_name)
            )
        )
        target_net.load_state_dict(torch.load(
            '%s/checkpoint/%s-temp.pth' %(save_root, target_dataset_name)
            )
        )
        alpha_change_point_file.write('%s\n' %str(niter))

    optimizer_s = optim.Adam(source_net.parameters(), lr=1e-3)
    optimizer_t = optim.Adam(target_net.parameters(), lr=1e-3)

    ###############
    # Co-Training #
    ###############
    small_train_loss = 1e9
    descend_count = 0
    stop_flag = False
    best_test_acc = 0

    for epoch in range(n_epoch):

        if stop_flag : break

        print('\nEpoch: %d' %epoch)
        source_net.train()
        target_net.train()
        end = time.time()

        total = 0
        correct_t = 0
        correct_s = 0
        loss_t = 0

        for batch_idx, ((x_s, y_s), (x_t, y_t)) \
                in enumerate(zip(source_loader, target_loader)):

            niter += 1 # Iteration number for indexing

            if use_cuda:
                x_s, y_s, x_t, y_t = \
                    x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()

            #########################
            # Co-Prune with Weights #
            #########################
            # if epoch == 0 and batch_idx == 0:
            for layer_name, CR in CR_ratio.items():

                source_layer = getattr(source_net, layer_name)
                target_layer = getattr(target_net, layer_name)
                source_weight = torch.abs(source_layer.weight.data)
                target_weight = torch.abs(target_layer.weight.data)
                n_left = int(source_weight.numel() * CR)

                sumup_weight = alpha * source_weight + (1 - alpha) * target_weight
                domain_invariant_thresh = torch.topk(sumup_weight.view(-1), n_left)[0][-1]
                domain_invariant_mask = (sumup_weight > domain_invariant_thresh)
                target_mask_dict[layer_name] = domain_invariant_mask.float().cuda()

                source_thresh = torch.topk(source_weight.view(-1), n_left)[0][-1]
                source_mask = (source_weight > source_thresh)
                source_mask_dict[layer_name] = source_mask.float()

            # print('>>>>>>>>>>>>>>>>> Mask update <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            optimizer_s.zero_grad()
            optimizer_t.zero_grad()

            out_s = source_net(x_s, source_mask_dict)
            out_t = target_net(x_t, target_mask_dict)
            losses_s = nn.CrossEntropyLoss()(out_s, y_s)
            losses_t = nn.CrossEntropyLoss()(out_t, y_t)
            # losses = losses_s + losses_t
            # losses.backward()
            losses_s.backward()
            losses_t.backward()
            optimizer_s.step()
            optimizer_t.step()

            _, predicted = torch.max(out_s.data, dim=1)
            correct_s += predicted.eq(y_s.data).cpu().sum().item()
            _, predicted = torch.max(out_t.data, dim=1)
            correct_t += predicted.eq(y_t.data).cpu().sum().item()

            loss_t += losses_t.item()

            total += y_s.size(0)

            progress_bar(batch_idx, min(len(source_loader), len(target_loader)), "[Training] Source acc: %.3f%% | Target acc: %.3f%%"
                             %(100.0 * correct_s / total, 100.0 * correct_t / total))

            #######################
            # Record Training log #
            #######################
            source_recorder.update(loss=losses_s.item(), acc=accuracy(out_s.data, y_s.data, (1, 5)),
                            batch_size=out_s.shape[0], cur_lr=optimizer_s.param_groups[0]['lr'], end=end)

            target_recorder.update(loss=losses_t.item(), acc=accuracy(out_t.data, y_t.data, (1, 5)),
                                   batch_size=out_t.shape[0], cur_lr=optimizer_t.param_groups[0]['lr'], end=end)

        # Test target acc
        test_acc = mask_test(target_net, target_mask_dict, target_test_loader)
        print('\n[Epoch %d] Test Acc: %.3f' % (epoch, test_acc))
        target_recorder.update(loss=None, acc=test_acc, batch_size=0, end=None, is_train=False)

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            if not os.path.isdir('%s/checkpoint' %save_root):
                os.makedirs('%s/checkpoint' %save_root)
            torch.save(source_net.state_dict(), '%s/checkpoint/%s-temp.pth' %(save_root, source_dataset_name))
            torch.save(target_net.state_dict(), '%s/checkpoint/%s-temp.pth' %(save_root, target_dataset_name))

        if loss_t < small_train_loss:
            small_train_loss = loss_t
            descend_count = 0

        else:
            descend_count += 1

        print('\nTraining loss: %.3f, descend count: %d' % (loss_t, descend_count))

        if descend_count >= 3:

            descend_count = 0
            optimizer_t.param_groups[0]['lr'] *= 0.1
            optimizer_s.param_groups[0]['lr'] *= 0.1
            print('Learning rate: %e' % optimizer_t.param_groups[0]['lr'])
            if optimizer_t.param_groups[0]['lr'] <= 1e-6:
                stop_flag = True
                break

    print('Best test acc: %.3f' % best_test_acc)
    best_acc_list.append(best_test_acc)

print(best_acc_list)
source_recorder.close()
target_recorder.close()
alpha_change_point_file.close()