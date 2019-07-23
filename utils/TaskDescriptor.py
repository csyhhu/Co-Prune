"""
A class to incorprate all tasks
"""

import torch
import torch.optim as optim

import numpy as np
import random
import time
import os
import shutil

# from utils.MetaPruner import retrieve_kernel_info, retrieve_features_dim
from utils.dataset import get_dataloader, get_lmdb_imagenet
from utils.train import accuracy, AverageMeter, progress_bar
from meta_utils.adam import Adam
from meta_utils.SGD import SGD
from models_CIFAR.sparse_meta_resnet import resnet20_cifar, resnet20_stl, \
    resnet32_stl, resnet32_cifar, resnet56_cifar, resnet56_stl
from models_CIFAR.soft_quantized_resnet import resnet20_cifar as soft_quantized_resnet20_cifar, \
    resnet20_stl as soft_quantized_resnet20_stl
from models_CIFAR.sparse_meta_vgg import vgg11, vgg11_stl10
from models_ImageNet.sparse_meta_resnet import resnet18
from utils.recorder import Recorder


def yielder(dataloader):

    while True:
        for inputs, targets in dataloader:
            yield inputs, targets

class Task():

    def __init__(self, task_name, task_type = 'prune', optimizer_type = 'adam',
                 save_root = None, SummaryPath = None, use_cuda = True, **kwargs):

        self.task_name = task_name
        self.task_type = task_type # prune, soft-quantize
        self.model_name, self.dataset_name = task_name.split('-')
        self.ratio = 'sample' if self.dataset_name in ['CIFARS'] else -1

        #######
        # Net #
        #######
        if task_type == 'prune':
            if self.model_name == 'ResNet20':
                if self.dataset_name in ['CIFAR10', 'CIFARS']:
                    self.net = resnet20_cifar()
                elif self.dataset_name == 'STL10':
                    self.net = resnet20_stl()
                else:
                    raise NotImplementedError
            elif self.model_name == 'ResNet32':
                if self.dataset_name in ['CIFAR10', 'CIFARS']:
                    self.net = resnet32_cifar()
                elif self.dataset_name == 'STL10':
                    self.net = resnet32_stl()
                else:
                    raise NotImplementedError
            elif self.model_name == 'ResNet56':
                if self.dataset_name in ['CIFAR10', 'CIFARS']:
                    self.net = resnet56_cifar()
                elif self.dataset_name == 'CIFAR100':
                    self.net = resnet56_cifar(num_classes=100)
                elif self.dataset_name == 'STL10':
                    self.net = resnet56_stl()
                else:
                    raise NotImplementedError
            elif self.model_name == 'ResNet18':
                if self.dataset_name == 'ImageNet':
                    self.net = resnet18()
                else:
                    raise NotImplementedError
            elif self.model_name == 'vgg11':
                self.net = vgg11() if self.dataset_name == 'CIFAR10' else vgg11_stl10()
            else:
                print(self.model_name, self.dataset_name)
                raise NotImplementedError
        elif task_type == 'soft-quantize':
            if self.model_name == 'ResNet20':
                if self.dataset_name in ['CIFAR10', 'CIFARS']:
                    self.net = soft_quantized_resnet20_cifar()
                elif self.dataset_name in ['STL10']:
                    self.net = soft_quantized_resnet20_stl()
            else:
                raise NotImplementedError
        else:
            raise ('Task type not defined.')


        self.meta_opt_flag = True # True for enabling meta leraning

        ##############
        # Meta Prune #
        ##############
        self.mask_dict = dict()
        self.meta_grad_dict = dict()
        self.meta_hidden_state_dict = dict()

        ######################
        # Meta Soft Quantize #
        ######################
        self.quantized = 0 # Quantized type
        self.alpha_dict = dict()
        self.alpha_hidden_dict = dict()
        self.sq_rate = 0
        self.s_rate = 0
        self.q_rate = 0

        ##########
        # Record #
        ##########
        self.dataset_type = 'large' if self.dataset_name in ['ImageNet'] else 'small'
        self.SummaryPath = SummaryPath
        self.save_root = save_root

        self.recorder = Recorder(self.SummaryPath, self.dataset_name, self.task_name)

        ####################
        # Load Pre-trained #
        ####################
        self.pretrain_path = '%s/%s-pretrain.pth' %(self.save_root, self.task_name)
        self.net.load_state_dict(torch.load(self.pretrain_path))
        print('Load pre-trained model from %s' %self.pretrain_path)

        if use_cuda:
            self.net.cuda()

        # Optimizer for this task
        if optimizer_type in ['Adam', 'adam']:
            self.optimizer = Adam(self.net.parameters(), lr=1e-3)
        else:
            self.optimizer = SGD(self.net.parameters())

        if self.dataset_name == 'ImageNet':
            try:
                self.train_loader = get_lmdb_imagenet('train', 128)
                self.test_loader = get_lmdb_imagenet('test', 100)
            except:
                self.train_loader = get_dataloader(self.dataset_name, 'train', 128)
                self.test_loader = get_dataloader(self.dataset_name, 'test', 100)
        else:
            self.train_loader = get_dataloader(self.dataset_name, 'train', 128, ratio=self.ratio)
            self.test_loader = get_dataloader(self.dataset_name, 'test', 128)

        self.iter_train_loader = yielder(self.train_loader)
        # For shared
        # self.loss = 0
        # self.niter = 0 # Overall iteration record
        # self.test_loss = 0
        # self.smallest_training_loss = 1e9
        # self.stop = False # Whether to stop training
        #
        # # For CIFAR dataset
        # # self.train_acc = AverageMeter()
        # self.total = 0 # Number of batches used in training
        # self.n_batch = 0 # Number of batches used in training
        # self.test_acc = 0
        # self.best_test_acc = 0
        # self.ascend_count = 0
        #
        # # For ImageNet dataset
        # # self.loss = AverageMeter()
        # self.top1 = AverageMeter()
        # self.top5 = AverageMeter()
        # self.batch_time = AverageMeter()
        # self.data_time = AverageMeter()
        # self.test_acc_top1 = 0
        # self.test_acc_top5 = 0
        # self.best_test_acc_top1 = 0
        # self.best_test_acc_top5 = 0
        #
        # #######################
        # # Parameters for Meta #
        # #######################
        # self.mask_dict = dict()
        # self.meta_grad_dict = dict()
        # self.meta_hidden_state_dict = dict()
        #
        # ###########################
        # # Open File for Recording #
        # ###########################
        # if self.dataset_type == 'small':
        #     self.loss_record = open('%s/%s-loss.txt' %(self.SummaryPath, self.task_name), 'w+')
        #     self.train_acc_record = open('%s/%s-train-acc.txt' %(self.SummaryPath, self.task_name), 'w+')
        #     self.test_acc_record = open('%s/%s-test-acc.txt' %(self.SummaryPath, self.task_name), 'w+')
        #     self.lr_record = open('%s/%s-lr.txt' %(self.SummaryPath, self.task_name), 'w+')
        #     # print('Initialize %s' %(self.task_name))
        # else:
        #     self.loss_record = open('%s/%s-loss.txt' % (self.SummaryPath, self.task_name), 'w+')
        #     self.train_top1_acc_record = open('%s/%s-train-top1-acc.txt' % (self.SummaryPath, self.task_name), 'w+')
        #     self.train_top5_acc_record = open('%s/%s-train-top5-acc.txt' % (self.SummaryPath, self.task_name), 'w+')
        #     self.test_top1_acc_record = open('%s/%s-test-top1-acc.txt' % (self.SummaryPath, self.task_name), 'w+')
        #     self.test_top5_acc_record = open('%s/%s-test-top5-acc.txt' % (self.SummaryPath, self.task_name), 'w+')
        #     self.lr_record = open('%s/%s-lr.txt' % (self.SummaryPath, self.task_name), 'w+')

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def update_record_performance(self, loss, acc, batch_size=0, lr = 1e-3, end=None, is_train = True):

        self.recorder.update(loss=loss, acc=acc, batch_size=batch_size, cur_lr=lr, end=end, is_train=is_train)

        # if is_train:
        #
        #     self.loss += loss
        #     self.n_batch += 1
        #     self.total += batch_size
        #     self.niter += 1
        #
        #     if self.dataset_type == 'small':
        #         self.top1.update(acc[0], batch_size)
        #
        #         self.loss_record.write('%d, %.8f\n' % (self.niter, self.loss / self.n_batch))
        #         self.train_acc_record.write('%d, %.3f\n' % (self.niter, self.top1.avg))
        #         self.lr_record.write('%d, %e\n' % (self.niter, self.optimizer.param_groups[0]['lr']))
        #
        #         self.flush([self.loss_record, self.train_acc_record, self.lr_record])
        #
        #     else:
        #         self.batch_time.update(time.time() - end)
        #         self.top1.update(acc[0], batch_size)
        #         self.top5.update(acc[1], batch_size)
        #
        #         self.loss_record.write('%d, %.8f\n' % (self.niter, self.loss / self.n_batch))
        #         self.train_top1_acc_record.write('%d, %.3f\n' % (self.niter, self.top1.avg))
        #         self.train_top5_acc_record.write('%d, %.3f\n' % (self.niter, self.top5.avg))
        #         self.lr_record.write('%d, %ef\n' % (self.niter, self.optimizer.param_groups[0]['lr']))
        #
        #         self.flush([self.loss_record, self.train_top1_acc_record, self.train_top5_acc_record, self.lr_record])
        #
        # else:
        #     self.test_loss = loss
        #
        #     if self.dataset_type == 'small':
        #
        #         self.test_acc = acc
        #
        #         if self.best_test_acc < self.test_acc:
        #             self.best_test_acc = self.test_acc
        #             print('[%s] Best test acc' %self.task_name)
        #             # self.save(self.SummaryPath)
        #
        #         self.test_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc))
        #         self.flush([self.test_acc_record])
        #
        #     else:
        #
        #         self.test_acc_top1, self.test_acc_top5 = acc[0], acc[1]
        #
        #         if self.best_test_acc_top1 < self.test_acc_top1 or self.best_test_acc_top5 < self.test_acc_top5:
        #             self.best_test_acc_top1 = self.test_acc_top1
        #             self.best_test_acc_top5 = self.test_acc_top5
        #             print('[%s] Best test acc' % self.task_name)
        #             # self.save(self.SummaryPath)
        #
        #         self.test_top1_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc_top1))
        #         self.test_top5_acc_record.write('%d, %.3f\n' % (self.niter, self.test_acc_top5))
        #
        #         self.flush([self.test_top1_acc_record, self.test_top5_acc_record])


    def reset_performance(self):

        # self.loss = 0
        #
        # if self.dataset_type == 'small':
        #     self.loss = 0
        #     # self.train_acc.reset()
        #     self.top1.reset()
        #     self.total = 0
        #     self.n_batch = 0
        # else:
        #     self.best_test_acc_top1 = 0
        #     self.best_test_acc_top5 = 0
        #     self.top1.reset()
        #     self.top5.reset()
        #     self.batch_time.reset()
        self.recorder.reset_performance()


    # def set_best_acc(self, test_acc):
    #     self.best_test_acc = test_acc


    def save(self, save_root):
        torch.save(self.net.state_dict(), '%s/%s-net.pth' %(save_root, self.task_name))

    def get_best_test_acc(self):

        # if self.dataset_type == 'small':
        #     return self.best_test_acc
        # else:
        #     return self.best_test_acc_top1, self.best_test_acc_top5
        return self.recorder.get_best_test_acc()

    def flush(self, file_list=None):

        for file in file_list:
            file.flush()

    def close(self):

        # if self.dataset_type == 'small':
        #     self.loss_record.close()
        #     self.train_acc_record.close()
        #     self.test_acc_record.close()
        #     self.lr_record.close()
        # else:
        #     self.loss_record.close()
        #     self.train_top1_acc_record.close()
        #     self.train_top5_acc_record.close()
        #     self.test_top1_acc_record.close()
        #     self.test_top5_acc_record.close()
        #     self.lr_record.close()
        self.recorder.close()

    def adjust_lr(self, adjust_type):

        # if self.dataset_type == 'small':
        #     if self.loss > self.smallest_training_loss:
        #         self.ascend_count += 1
        #     else:
        #         self.smallest_training_loss = self.loss
        #         self.ascend_count = 0
        #
        #     if self.ascend_count >= 3:
        #         self.ascend_count = 0
        #         self.optimizer.param_groups[0]['lr'] *= 0.1
        #         if self.optimizer.param_groups[0]['lr'] < 1e-6:
        #             self.stop = True
        #
        #     print('[%s] Current training loss: %.3f[%.3f], ascend count: %d'
        #           %(self.task_name, self.loss, self.smallest_training_loss, self.ascend_count))
        #     print('---------------------------------------------------')
        # else:
        #     raise NotImplementedError

        self.recorder.adjust_lr(self.optimizer)



class TaskDescriptor():

    def __init__(self, task_name_list, task_type = 'prune', save_root = None, SummaryPath = None, use_cuda = True):

        if os.path.exists(SummaryPath):
            print('Summary folder exist, remove?')
            # input()
            shutil.rmtree(SummaryPath)
            os.makedirs(SummaryPath)
        else:
            os.makedirs(SummaryPath)

        self.task_dict = dict()

        for task_name in task_name_list:
            self.task_dict[task_name] = Task(task_name=task_name,
                                             save_root=save_root, SummaryPath=SummaryPath,
                                             use_cuda=use_cuda, task_type=task_type)
            print('Initialize task %s finish.' %task_name)

    def train(self):
        for task in self.task_dict.values():
            task.train()

    def eval(self):
        for task in self.task_dict.values():
            task.eval()

    def sample(self):
        return random.sample(self.task_dict.items(), k=1)[0]

    def initialize_record(self, SummaryRoot):
        pass

    def save(self, save_root):
        for task_name, task in self.task_dict.items():
            task.save(save_root)


if __name__ == '__main__':

    task_list = ['ResNet20-STL10']
    taskDescriptor = TaskDescriptor(task_list)

    for idx in range(10):
        selected_task_name, selected_task = taskDescriptor.sample()
        # (inputs, targets) = taskDescriptor.fetch_data(selected_task_name)
        inputs, targets = selected_task.train_loader.__next__()
        print(inputs.shape)

