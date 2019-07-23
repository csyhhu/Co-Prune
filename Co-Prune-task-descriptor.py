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
from utils.train import progress_bar
from utils.sparse import mask_test
from utils.recorder import Recorder
from utils.TaskDescriptor import TaskDescriptor

import itertools
import pickle
import shutil
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Co-Prune')
parser.add_argument('--model', '-m', type=str, default='CIFARNet', help='Model')
parser.add_argument('--srctgt', '-st', nargs='+', default='CIFAR9 STL9', help='Source and target domain')
parser.add_argument('--exp_spec', '-e', type=str, default='', help='Experiment Specification')
parser.add_argument('--n_epoch', '-n', type=int, default=100, help='Maximum training epochs')
parser.add_argument('--alpha','-a', nargs='+', help='Variation of alpha', default="0.7 0.5 0.3", required=False)
args = parser.parse_args()

# ---------------------------------------------------------------
use_cuda = torch.cuda.is_available()
model_name = args.model
source_dataset_name = args.srctgt[0]
target_dataset_name = args.srctgt[1]
n_epoch = args.n_epoch
exp_spec = args.exp_spec
alpha_list = args.alpha
# ----------------------------------------------------------------

# Specify CR
from CR_setting import CR_ratio_1_35 as CR_ratio, get_sparse_param_analysis
CR_list = [CR_ratio] * len(alpha_list)


#########################
# Load Pretrained Model #
#########################
# source_net = SparseCIFARNet()
# target_net = SparseCIFARNet()
# source_pretrain_param = torch.load(source_pretrain_path)
# target_pretrain_param = torch.load(target_pretrain_path)
#
# source_net.load_state_dict(source_pretrain_param)
# target_net.load_state_dict(target_pretrain_param)

# overall_CR = get_sparse_param_analysis(target_net, CR_ratio)

# if use_cuda:
#     source_net.cuda()
#     target_net.cuda()

#####################
# Initial mask dict #
#####################
source_mask_dict = dict()
target_mask_dict = dict()

####################
# Initial Recorder #
####################
save_root = './Results/%s' %(model_name)
task_name_list = []
for domain_idx, domain_name in enumerate(args.srctgt):
    save_root += '%s' %(domain_name)
    task_name_list.append('%s-%s' %(model_name, domain_name))
    if domain_idx < len(args.srctgt)-1:
        save_root += '-'

SummaryPath = '%s/runs/CR-%.2f' %(save_root, overall_CR)
if args.exp_spec is not '':
    SummaryPath += ('-' + args.exp_spec)

print('Save to %s' %SummaryPath)

taskDescriptor = TaskDescriptor(task_name_list, save_root=save_root, SummaryPath=SummaryPath)

source_loader = taskDescriptor.task_dict['%s-%s' %(model_name, source_dataset_name)].train_loader
target_loader = taskDescriptor.task_dict['%s-%s' %(model_name, source_dataset_name)].train_loader

best_acc_list = []
niter = 0
for ite, CR_ratio in enumerate(CR_list):

    alpha = alpha_list[ite]

    print('Adaptive iteration: %d, alpha: %.3f' % (ite, alpha))
    print('Current CR: %s' % CR_ratio)
    print('niter: %d' %niter)

    if ite != 0:
        print('Loading best model from previous alpha')
        source_net.load_state_dict(torch.load(
            '%s/checkpoint/source-temp.pth' %(save_root)
            )
        )
        target_net.load_state_dict(torch.load(
            '%s/checkpoint/target-temp.pth' %(save_root)
            )
        )

    ########################
    # Initialize Optimizer #
    ########################
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
            if epoch == 0 and batch_idx == 0:
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
                print('>>>>>>>>>>>>>>>>> Mask update <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

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

            # if writer is not None:
            if SummaryPath is not None:
                writer.add_scalar('Train/Accuracy', 100.0 * correct_t / total, niter)
                writer.add_scalar('Train/Loss', loss_t / (batch_idx + 1), niter)

                loss_record.write('%d, %.3f\n' % (niter, loss_t / (batch_idx + 1)))
                train_acc_record.write('%d, %.3f\n' % (niter, 100. * float(correct_t) / float(total)))
                lr_record.write('%d, %e\n' % (niter, optimizer_t.param_groups[0]['lr']))


        # print('CR: %.3f%% | %.3f%% | %.3f%% | %.3f%% | %.3f%%' %(
        #     100.0 * torch.sum(target_mask_dict['conv1']) / target_mask_dict['conv1'].numel(),
        #     100.0 * torch.sum(target_mask_dict['conv2']) / target_mask_dict['conv2'].numel(),
        #     100.0 * torch.sum(target_mask_dict['conv3']) / target_mask_dict['conv3'].numel(),
        #     100.0 * torch.sum(target_mask_dict['fc1']) / target_mask_dict['fc1'].numel(),
        #     100.0 * torch.sum(target_mask_dict['fc2']) / target_mask_dict['fc2'].numel())
        # )
        test_acc = mask_test(target_net, target_mask_dict, target_test_loader)
        print('\n[Epoch %d] Test Acc: %.3f' % (epoch, test_acc))
        # if writer is not None:
        if SummaryPath is not None:
            writer.add_scalar('Test/acc', test_acc, niter)
            test_acc_record.write('%d, %.3f\n' % (niter, test_acc))

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            # torch.save(source_net.state_dict(), './checkpoint/ada-co-prune-s.pth')
            # torch.save(target_net.state_dict(), './checkpoint/ada-co-prune-t.pth')
            # torch.save(source_net.state_dict(), './checkpoint/adaptive/co-prune-s-%.2f.pth' %(alpha))
            # torch.save(target_net.state_dict(), './checkpoint/adaptive/co-prune-t-%.2f.pth' %(alpha))
            # pickle.dump(source_mask_dict, open('./checkpoint/adaptive/co-prune-mask-s-%.1f.pkl' %(alpha), 'wb'))
            # pickle.dump(target_mask_dict, open('./checkpoint/adaptive/co-prune-mask-t-%.1f.pkl' %(alpha), 'wb'))
            torch.save(source_net.state_dict(), './checkpoint/adaptive/co-prune-s-baseline-one-time.pth')
            torch.save(target_net.state_dict(), './checkpoint/adaptive/co-prune-t-baseline-one-time.pth')

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
            print('Learning rata: %e' % optimizer_t.param_groups[0]['lr'])
            if optimizer_t.param_groups[0]['lr'] <= 1e-6:
                stop_flag = True
                break

    print('Best test acc: %.3f' % best_test_acc)
    best_acc_list.append(best_test_acc)

print(best_acc_list)
if writer is not None:
    writer.close()