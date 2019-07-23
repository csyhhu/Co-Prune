from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.cifar import CIFAR10
import pickle
from operator import itemgetter


class STL9(CIFAR10):
    """`STL9 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    '''
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
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
    '''
    splits = ('train', 'test', 'limited', 'fake', 'fake_train')
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, resample=False, n_per_classes=20):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train/test/
        self.n_pre_classes = n_per_classes

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError(
        #         'Dataset not found or corrupted. '
        #         'You can use download=True to download it')

        # now load the picked numpy arrays
        if (split == 'limited' and not os.path.exists('%s/%s' %(self.root, self.split))) or resample:
            self.generate_limited_dataset()
        data_collection = pickle.load(open('%s/%s' %(self.root, self.split), 'rb'))

        self.data = data_collection['data']
        if split == 'fake':
            self.data = (np.transpose(self.data, (0, 2, 3, 1)) * 255.0).astype(np.uint8)
        if split == 'fake_train':
            self.data = (self.data * 255.0).astype(np.uint8)
        self.labels = data_collection['labels']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    '''
    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels
    '''

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


    def generate_limited_dataset(self):
        data_collection = pickle.load(open('%s/%s' % (self.root, 'train'), 'rb'))
        self.data = data_collection['data']
        self.labels = data_collection['labels']
        # Number of instances for each classes
        n_classes = len(np.unique(self.labels))
        count_n_classes = np.zeros([n_classes])

        indices = np.random.permutation(self.data.shape[0])
        selected_indices = []

        for idx in indices:
            cls = self.labels[idx]
            if count_n_classes[cls] < self.n_pre_classes:
                count_n_classes[cls] += 1
                selected_indices.append(idx)
            if len(selected_indices) == n_classes * self.n_pre_classes:
                print('Select enough data')
                break

        # Slice training data and labels into limited counterparts
        limited_data = self.data[selected_indices]
        limited_labels = itemgetter(*selected_indices)(self.labels)

        print('Random sample limited training data size: %d. Saving...' \
              % (limited_data.shape[0]))
        pickle.dump({'data': limited_data, 'labels': limited_labels},
                    open(os.path.join(self.root, 'limited'), 'wb'))

if __name__ == '__main__':

    dataset = STL9('/data/datasets/STL9', split='limited')