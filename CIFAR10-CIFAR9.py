"""
This code regenerate CIFAR10/STL10 dataset to align them:
Delete frog(6) from CIFAR10 and monkey(7) from STL10.
STL10 labels map to align CIFAR10
"""

from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
# from .utils import download_url, check_integrity
from torchvision.datasets.utils import download_url, check_integrity
from operator import itemgetter

# -------------
# Load CIFAR10
# -------------

train_data = []
train_labels = []
train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
]

test_data = []
test_labels = []
test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
]

dataset_root = '/home/shangyu/datasets' # Set up your own dataset path

for fentry in train_list:
    f = fentry[0]
    file = os.path.join('%s/CIFAR10/cifar-10-batches-py' %dataset_root, f)
    fo = open(file, 'rb')
    if sys.version_info[0] == 2:
        entry = pickle.load(fo)
    else:
        entry = pickle.load(fo, encoding='latin1')
    train_data.append(entry['data'])
    if 'labels' in entry:
        train_labels += entry['labels']
    else:
        train_labels += entry['fine_labels']
    fo.close()

train_data = np.concatenate(train_data)
train_data = train_data.reshape((50000, 3, 32, 32))
train_data = train_data.transpose((0, 2, 3, 1))

f = test_list[0][0]
file = os.path.join('%s/CIFAR10/cifar-10-batches-py' %dataset_root, f)
fo = open(file, 'rb')
if sys.version_info[0] == 2:
    entry = pickle.load(fo)
else:
    entry = pickle.load(fo, encoding='latin1')
test_data = entry['data']
if 'labels' in entry:
    test_labels = entry['labels']
else:
    test_labels = entry['fine_labels']
fo.close()
test_data = test_data.reshape((10000, 3, 32, 32))
test_data = test_data.transpose((0, 2, 3, 1))

# Remove frog(6) and minus 1 after 6
keep_train_data = []
keep_train_labels = []
for idx in range(train_data.shape[0]):
    if train_labels[idx] == 6:
        continue
    elif train_labels[idx] > 6:
        keep_train_data.append(train_data[idx,:].reshape(-1, 32, 32, 3))
        keep_train_labels.append(train_labels[idx]-1)
    else:
        keep_train_data.append(train_data[idx, :].reshape(-1, 32, 32, 3))
        keep_train_labels.append(train_labels[idx])

keep_train_data = np.concatenate(keep_train_data)

keep_test_data = []
keep_test_labels = []
for idx in range(test_data.shape[0]):
    if test_labels[idx] == 6:
        continue
    elif test_labels[idx] > 6:
        keep_test_data.append(test_data[idx,:].reshape(-1, 32, 32, 3))
        keep_test_labels.append(test_labels[idx]-1)
    else:
        keep_test_data.append(test_data[idx, :].reshape(-1, 32, 32, 3))
        keep_test_labels.append(test_labels[idx])

keep_test_data = np.concatenate(keep_test_data)

if not os.path.exists('%s/CIFAR9' %dataset_root):
    os.makedirs('%s/CIFAR9' %dataset_root)

pickle.dump({'data': keep_train_data, 'labels': keep_train_labels},
    open('%s/CIFAR9/train' %dataset_root, 'wb'))

pickle.dump({'data': keep_test_data, 'labels': keep_test_labels},
    open('%s/CIFAR9/test' %dataset_root, 'wb'))