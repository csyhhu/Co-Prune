"""
Some helper functions and classes to implement a sparse network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import progress_bar

class Function_STE_sparse(torch.autograd.Function):
    """
    SET Sparse conducts forward by masking the gate matrix into the weights, during the backward, it permeates
    gradient through the zero elements in gate matrix
    """

    @staticmethod
    def forward(ctx, weight, mask):
        ctx.save_for_backward(weight, mask)
        return weight * mask

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, mask = ctx.saved_tensors
        grad_weight = grad_outputs.clone()
        grad_mask = grad_outputs * weight
        return grad_weight, grad_mask


class sparse_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(sparse_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.sparse_weight = None

    def forward(self, input, mask = None, sparse_type = 'STE'):

        if mask is not None:
            if sparse_type == 'direct':
                self.sparse_weight = self.weight * mask
            elif sparse_type == 'STE':
                self.sparse_weight = Function_STE_sparse.apply(self.weight, mask)
            else:
                raise NotImplementedError
            return F.conv2d(input, self.sparse_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class sparse_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(sparse_Linear, self).__init__(in_features, out_features, bias=bias)

        self.sparse_weight = None

    def forward(self, input, mask = None, sparse_type = 'STE'):

        if mask is not None:
            if sparse_type == 'direct':
                self.sparse_weight = self.weight * mask
            elif sparse_type == 'STE':
                self.sparse_weight = Function_STE_sparse.apply(self.weight, mask)
            else:
                raise NotImplementedError
            return F.linear(input, self.sparse_weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


class Function_DNS(torch.autograd.Function):
    """
    DNS Pruning conducts forward by masking the gate matrix into the weights, during the backward, it permeates
    gradient through the zero elements in gate matrix
    """
    @staticmethod
    def forward(ctx, weight, CR):
        # CR: Portion of elements saved
        n_left = int(weight.numel() * CR)
        thresh = torch.topk(torch.abs(weight).view(-1), k=n_left)[0][-1]
        mask = (torch.abs(weight) > thresh).float()
        return weight * mask

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = grad_outputs.clone()
        return grad_inputs, None


class DNS_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(DNS_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.sparse_weight = None
        # self.mask = None

    def forward(self, input, CR = None):

        if CR is not None:
            self.sparse_weight = Function_DNS.apply(self.weight, CR)
            return F.conv2d(input, self.sparse_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


class DNS_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(DNS_Linear, self).__init__(in_features, out_features, bias=bias)

        self.sparse_weight = None

    def forward(self, input, CR = None):

        if CR is not None:
            self.sparse_weight = Function_DNS.apply(self.weight, CR)
            return F.linear(input, self.sparse_weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)




def static_mask_train(net, mask_dict, optimizer, train_loader, validate_loader, max_epoch = 100,
                      criterion = nn.CrossEntropyLoss(), save_path = None, min_lr = 1e-6,
                      max_descent_count=3, use_cuda = True):

    small_train_loss = 1e9
    descend_count = 0
    stop_flag = False
    best_test_acc = 0

    for epoch in range(max_epoch):

        if stop_flag: break

        net.train()
        total = 0
        correct = 0
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs, mask_dict)
            losses = criterion(outputs, targets)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            total += targets.size(0)

            # progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%%"
            #              % (train_loss / (batch_idx + 1), 100.0 * correct / total))

        test_acc = mask_test(net, mask_dict, validate_loader)

        progress_bar(epoch, max_epoch, "Acc: %.3f%%" % test_acc)

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            if save_path is not None:
                # print('Saving...')
                torch.save(net.state_dict(), save_path)

        if train_loss < small_train_loss:
            small_train_loss = train_loss
            descend_count = 0

        else:
            descend_count += 1

        # print('Training loss: %.3f, descend count: %d' % (train_loss, descend_count))

        if descend_count >= max_descent_count:
            descend_count = 0
            optimizer.param_groups[0]['lr'] *= 0.1
            # print('Learning rata: %e' % optimizer.param_groups[0]['lr'])
            if optimizer.param_groups[0]['lr'] < min_lr:
                stop_flag = True
                print('\nBest acc: %.3f' %best_test_acc)
                break

    return best_test_acc


def DNS_train(net, CR, optimizer, train_loader, validate_loader, max_epoch = 100,
                      criterion = nn.CrossEntropyLoss(), save_path = None, min_lr = 1e-6,
                      max_descent_count=3, use_cuda = True):

    small_train_loss = 1e9
    descend_count = 0
    stop_flag = False
    best_test_acc = 0

    for epoch in range(max_epoch):

        if stop_flag: break

        net.train()
        total = 0
        correct = 0
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs, CR)
            losses = criterion(outputs, targets)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            total += targets.size(0)

        test_acc = DNS_test(net, CR, validate_loader)

        progress_bar(epoch, max_epoch, "Acc: %.3f%%" % test_acc)

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            if save_path is not None:
                # print('Saving...')
                try:
                    torch.save(net.module.state_dict(), save_path)
                except:
                    torch.save(net.state_dict(), save_path)

        if train_loss < small_train_loss:
            small_train_loss = train_loss
            descend_count = 0

        else:
            descend_count += 1

        if descend_count >= max_descent_count:
            descend_count = 0
            optimizer.param_groups[0]['lr'] *= 0.1
            if optimizer.param_groups[0]['lr'] < min_lr:
                stop_flag = True
                print('\nBest acc: %.3f' %best_test_acc)
                break

    return best_test_acc


def mask_test(net, mask_dict, test_loader, use_cuda = True):

    correct = 0
    total = 0

    net.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, mask_dict)
        _, predicted = torch.max(outputs.data, dim=1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        total += targets.size(0)

        # progress_bar(batch_idx, len(test_loader), "Test accuracy: %.3f%% (%d/%d)"
        #              %(100.0 * correct / total, correct, total))

    return 100.0 * correct / total


def DNS_test(net, CR, test_loader, use_cuda = True):

    correct = 0
    total = 0

    net.eval()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, CR)
        _, predicted = torch.max(outputs.data, dim=1)
        correct += predicted.eq(targets.data).cpu().sum().item()
        total += targets.size(0)

    return 100.0 * correct / total




if __name__ == '__main__':

    pass