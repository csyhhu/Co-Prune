from PIL import Image
import os
import os.path
import numpy as np
import sys
import pickle
import torch
import torch.utils.data as data
import random
from random import sample

class mean_variance(object):

    def __init__(self):
        self.mean = {
            'b': [0.5246, 0.5139, 0.4830],
            'c': [0.5327, 0.5246, 0.4938],
            'i': [0.4688, 0.4593, 0.4328],
            'p': [0.4634, 0.4557, 0.4319]
        }
        self.variance = {
            'b': [0.2505, 0.2473, 0.2532],
            'c': [0.2545, 0.2505, 0.2521],
            'i': [0.2508, 0.2463, 0.2508],
            'p': [0.2355, 0.2330, 0.2397]
        }

    def get_mean_variance(self, split):
        return self.mean[split], self.variance[split]



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def default_target_transform(target):

    return torch.LongTensor([target])


class CLEF_dataset(data.Dataset):

    '''@property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels'''

    def __init__(self, root, split='b', \
                 transform=None, target_transform=None, add_split=None, ratio=-1, resample=False):
        self.root = root
        self.split = split

        if resample:
            # print('Resample %s into train (label and unlabel), test, continue?' %(split))
            # if not os.path.exists('%s/%s/image_list.pkl' % (root, split)):
            #     self.traverse_dataset(split)
            origin_dataset = pickle.load(open('%s/%s/image_list.pkl' % (root, split), 'rb'))

            n_train_per_class = int(50 * ratio)
            n_train_bin = np.zeros(12)
            train_list = []
            test_list = []
            random.shuffle(origin_dataset)
            for data in origin_dataset:
                label = int(data[1])
                if n_train_bin[label] < n_train_per_class:
                    n_train_bin[label] += 1
                    train_list.append(data)
                else:
                    test_list.append(data)
            pickle.dump(train_list, open('%s/%s/image_list_train.pkl' % (root, split), 'wb'))
            pickle.dump(test_list, open('%s/%s/image_list_test.pkl' % (root, split), 'wb'))
            print('[Resample] Number of instances in train list: %d, test list: %d' \
                  %(len(train_list), len(test_list)))
            # input('Continue?')
            '''
            n_train = int(len(origin_dataset) * 0.9)
            random.shuffle(origin_dataset)
            train_list = origin_dataset[0 : n_train]
            test_list = origin_dataset[n_train : ]
            pickle.dump(train_list, open('%s/%s/image_list_train.pkl' % (root, split), 'wb'))
            pickle.dump(test_list, open('%s/%s/image_list_test.pkl' % (root, split), 'wb'))
            n_label = int(0.5 * n_train)
            label_list = train_list[0: n_label]
            unlabel_list = train_list[n_label: ]
            pickle.dump(label_list, open('%s/%s/image_list_label.pkl' % (root, split), 'wb'))
            pickle.dump(unlabel_list, open('%s/%s/image_list_unlabel.pkl' % (root, split), 'wb'))
            '''

        # Load image list pickle
        if add_split is None:
            self.image_list = pickle.load(open('%s/%s/image_list.pkl' % (root, split), 'rb'))
        else:
            self.image_list = pickle.load(open('%s/%s/image_list_%s.pkl' % (root, split, add_split), 'rb'))
        if len(self.image_list) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root))

        # if ratio != -1:
        #     length = int(ratio * len(self.image_list))
        #     print('Randomly sample from dataset')
        #     self.image_list = sample(self.image_list, length)

        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        path, target = self.image_list[index]
        # print (target)
        sample = self.loader('%s/%s' %(self.root, path))
        # print (sample.size)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(int(target))
        target = int(target)
        # print(sample.shape)
        # print (target)
        # print('-----------------------------')
        return sample, target

    def __len__(self):
        return len(self.image_list)

    def traverse_dataset(self, split):
        pass


if __name__ == '__main__':

    pass