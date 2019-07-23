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
import matplotlib.pyplot as plt

import torch.utils.data as data
# from .utils import download_url, check_integrity
from torchvision.datasets.utils import download_url, check_integrity
from operator import itemgetter

def __loadfile(root, data_file, labels_file=None):

    base_folder = 'stl10_binary'

    labels = None
    if labels_file:
        path_to_labels = os.path.join(
            root, base_folder, labels_file)
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

    path_to_data = os.path.join(root, base_folder, data_file)
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 1, 3, 2))

    return images, labels

# ----------
# Load STL10
# ----------

STL2CIFAR = {
    0:0, 1:2, 2:1, 3:3, 4:4, 5:5, 6:6, 7:-1, 8:7, 9:8
}

train_list = [
    ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
    ['train_y.bin', '5a34089d4802c674881badbb80307741'],
    ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
]

test_list = [
    ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
    ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
]
splits = ('train', 'train+unlabeled', 'unlabeled', 'test')


dataset_root = '/home/shangyu/datasets'

train_data, train_labels = __loadfile(dataset_root + '/STL10', train_list[0][0], train_list[1][0])
test_data, test_labels = __loadfile(dataset_root + '/STL10', test_list[0][0], test_list[1][0])

# Remove monkey(7) can map everything:
keep_train_data = []
keep_train_labels = []
for idx in range(train_data.shape[0]):
    if train_labels[idx] == 7:
        continue
    else:
        # print(train_data[idx, :].shape)
        # plt.imshow(np.transpose(train_data[idx, :], (1, 2, 0)))
        # plt.show()
        # input()
        # data = train_data[idx, :].reshape(-1, 96, 96, 3)
        # keep_train_data.append(data)
        # plt.imshow(data[0,:])
        # plt.show()
        # ds
        data = np.transpose(train_data[idx, :], (1, 2, 0))
        # plt.imshow(data)
        # plt.show()
        # input()
        data = np.expand_dims(data, axis=0)
        keep_train_data.append(data)
        keep_train_labels.append(STL2CIFAR[train_labels[idx]])

# keep_train_data = np.concatenate(keep_train_data).astype('float')
keep_train_data = np.concatenate(keep_train_data)

keep_test_data = []
keep_test_labels = []
for idx in range(test_data.shape[0]):
    if test_labels[idx] == 7:
        continue
    else:
        # keep_test_data.append(test_data[idx, :].reshape(-1, 96, 96, 3))
        data = np.transpose(test_data[idx, :], (1, 2, 0))
        data = np.expand_dims(data, axis=0)
        keep_test_data.append(data)
        keep_test_labels.append(STL2CIFAR[test_labels[idx]])

# keep_test_data = np.concatenate(keep_test_data).astype('float')
keep_test_data = np.concatenate(keep_test_data)

if not os.path.exists('%s/STL9' %dataset_root):
    os.makedirs('%s/STL9' %dataset_root)

pickle.dump({'data': keep_train_data, 'labels': keep_train_labels},
    open('%s/STL9/train' %dataset_root, 'wb'))

pickle.dump({'data': keep_test_data, 'labels': keep_test_labels},
            open('%s/STL9/test' %dataset_root, 'wb'))