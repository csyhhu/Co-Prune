import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sparse import sparse_CNN, sparse_Linear, DNS_CNN, DNS_Linear

class CIFARNet(nn.Module):

    def __init__(self, num_classes=10):

        super(CIFARNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)  # 3 * 32 * 5 * 5 =
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)

        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3, 2))

        x = self.conv2(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = self.conv3(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SparseCIFARNet(nn.Module):

    def __init__(self, num_classes=10):

        super(SparseCIFARNet, self).__init__()

        self.conv1 = sparse_CNN(3, 32, 5, 1, 2)  # 3 * 32 * 5 * 5 =
        self.conv2 = sparse_CNN(32, 32, 5, 1, 2)
        self.conv3 = sparse_CNN(32, 64, 5, 1, 2)

        self.fc1 = sparse_Linear(576, 64)
        self.fc2 = sparse_Linear(64, num_classes)

    def forward(self, x, mask_dict = None):

        if mask_dict is not None:
            x = self.conv1(x, mask_dict['conv1'])
        else:
            x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3, 2))

        if mask_dict is not None:
            x = self.conv2(x, mask_dict['conv2'])
        else:
            x = self.conv2(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        if mask_dict is not None:
            x = self.conv3(x, mask_dict['conv3'])
        else:
            x = self.conv3(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = x.view(-1, 576)

        if mask_dict is not None:
            x = F.relu(self.fc1(x, mask_dict['fc1']))
        else:
            x = F.relu(self.fc1(x))

        if mask_dict is not None:
            x = self.fc2(x, mask_dict['fc2'])
        else:
            x = self.fc2(x)

        return x


class DNSCIFARNet(nn.Module):

    def __init__(self):

        super(DNSCIFARNet, self).__init__()

        self.conv1 = DNS_CNN(3, 32, 5, 1, 2)
        self.conv2 = DNS_CNN(32, 32, 5, 1, 2)
        self.conv3 = DNS_CNN(32, 64, 5, 1, 2)

        self.fc1 = DNS_Linear(576, 64)
        self.fc2 = DNS_Linear(64, 9)

    def forward(self, x, CR_list = None):

        if CR_list is not None:
            x = self.conv1(x, CR_list['conv1'])
        else:
            x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 3, 2))

        if CR_list is not None:
            x = self.conv2(x, CR_list['conv2'])
        else:
            x = self.conv2(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        if CR_list is not None:
            x = self.conv3(x, CR_list['conv3'])
        else:
            x = self.conv3(x)
        x = F.avg_pool2d(F.relu(x), 3, 2)

        x = x.view(-1, 576)

        if CR_list is not None:
            x = F.relu(self.fc1(x, CR_list['fc1']))
        else:
            x = F.relu(self.fc1(x))

        if CR_list is not None:
            x = self.fc2(x, CR_list['fc2'])
        else:
            x = self.fc2(x)

        return x


if __name__ == '__main__':

    net = CIFARNet()
    inputs = torch.rand([1, 3, 32, 32])
    outputs = net(inputs)
