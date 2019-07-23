"""
An utils code for loading dataset
"""
import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datetime import datetime
import utils.imagenet_utils as imagenet_utils
import utils.CIFAR10_utils as CIFAR10_utils
import utils.MNIST_utils as MNIST_utils
import utils.SVHN_utils as SVHN_utils
import utils.CLEF_utils as CLEF_utils
import utils.STL9_utils as STL9_utils
import utils.CIFAR9_utils as CIFAR9_utils


def get_mean_and_std(dataset, n_channels=3):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(n_channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_dataloader(dataset_name, split, batch_size, \
                   add_split = None, shuffle = True, ratio=-1, resample=False):

    print ('[%s] Loading %s-%s from %s' %(datetime.now(), split, add_split, dataset_name))

    if dataset_name == 'MNIST':

        data_root_list = ['/home/shangyu/MNIST', '/data/datasets/MNIST']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        # print(data_root)

        normalize = transforms.Normalize((0.1307,), (0.3081,))
        # if split == 'train':
        MNIST_transform =transforms.Compose([
                               # transforms.Resize(32),
                               transforms.ToTensor(),
                               normalize
                           ])
        # else:
        dataset = MNIST_utils.MNIST(data_root,
                                    train = True if split =='train' else False, add_split=add_split,
                                    download=True, transform=MNIST_transform, ratio=ratio)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    elif dataset_name == 'SVHN':

        data_root_list = ['/home/shangyu/SVHN', '/data/datasets/SVHN']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Data root found: %s' %data_root)
                break

        # normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        normalize = transforms.Normalize((0.4433,), (0.1192,))
        if split == 'train':
            trainset = SVHN_utils.SVHN(root=data_root, split='train', add_split=add_split, download=False,
                                     transform=transforms.Compose([
                                         transforms.Grayscale(),
                                         transforms.Resize(28),
                                         transforms.ToTensor(),
                                         normalize
                                         ]), ratio=ratio)
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        elif split == 'test':
            trainset = SVHN_utils.SVHN(root=data_root, split='test', download=False,
                                     transform=transforms.Compose([
                                         transforms.Grayscale(),
                                         transforms.Resize(28),
                                         transforms.ToTensor(),
                                         normalize
                                         ]))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        # return loader

    elif dataset_name == 'CIFAR10':

        data_root_list = ['/home/shangyu/CIFAR10', '/home/sinno/csy/CIFAR10', '/data/datasets/CIFAR10']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False,
                                                    # transform=transform_train)
            trainset = CIFAR10_utils.CIFAR10(root=data_root, train=True, download=True,
                                                    transform=transform_train, ratio=ratio)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    elif dataset_name == 'STL10':

        data_root_list = ['/data/datasets/STL10', '/home/shangyu/datasets/STL10']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':

            loader = torch.utils.data.DataLoader(
                datasets.STL10(
                    root=data_root, split='train', download=True,
                    transform=transforms.Compose([
                        # transforms.Pad(4),
                        # transforms.RandomCrop(96),
                        transforms.Resize(36),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize((0.4330, 0.4331, 0.4330), (0.2327, 0.2327, 0.2327)),
                    ])),
                batch_size=batch_size, shuffle=True)

        if split in ['test', 'val']:
            loader = torch.utils.data.DataLoader(
                datasets.STL10(
                    root=data_root, split='test', download=True,
                    transform=transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize((0.4330, 0.4331, 0.4330), (0.2327, 0.2327, 0.2327)),
                    ])),
                batch_size=batch_size, shuffle=False)

    elif dataset_name == 'CIFAR9':

        data_root_list = ['/home/shangyu/datasets/CIFAR9', '/data/datasets/CIFAR9']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        loader = torch.utils.data.DataLoader(
            CIFAR9_utils.CIFAR9(root=data_root, split=split, transform=transform, resample=resample),
             batch_size=batch_size, shuffle=shuffle, num_workers = 2)

        return loader


    elif dataset_name == 'STL9':

        data_root_list = ['/home/shangyu/datasets/STL9', '/data/datasets/STL9']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        # tensor([0.4330, 0.4331, 0.4330])
        # tensor([0.2327, 0.2327, 0.2327])

        if split in ['train', 'limited', 'fake']:
            transform = transforms.Compose([
                        transforms.Resize(36),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4330, 0.4331, 0.4330), (0.2327, 0.2327, 0.2327)),
                    ])
        elif split in ['fake_train']:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4330, 0.4331, 0.4330), (0.2327, 0.2327, 0.2327)),
            ])


        loader = torch.utils.data.DataLoader(
            STL9_utils.STL9(root=data_root, split=split, transform=transform, resample=resample),
             batch_size=batch_size, shuffle=shuffle, num_workers = 2)

        return loader


    elif dataset_name == 'ImageNet':
        data_root_list = ['/remote-imagenet', '/data/imagenet', '/mnt/public/imagenet']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        traindir = ('../train_imagenet_list.pkl', '../classes.pkl', '../classes-to-idx.pkl','%s/train' %data_root)
        valdir = ('../val_imagenet_list.pkl', '../classes.pkl', '../classes-to-idx.pkl', '%s/val-pytorch' %data_root)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split == 'train':
            trainDataset = imagenet_utils.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), ratio=ratio)
            print ('Number of training data used: %d' %(len(trainDataset)))
            loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

        elif split == 'val' or split == 'test':
            valDataset = imagenet_utils.ImageFolder(valdir, transforms.Compose([
                # transforms.CenterCrop(32),
		        # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            loader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

    elif dataset_name == 'Image_CLEF':

        data_root_list = ['/data/image_CLEF', '/home/shangyu/Image_CLEF', '/data/datasets/image_CLEF']

        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'b':
            normalize = transforms.Normalize(mean=[0.5246, 0.5139, 0.4830], std=[0.2505, 0.2473, 0.2532])
        elif split == 'c':
            normalize = transforms.Normalize(mean=[0.5327, 0.5246, 0.4938], std=[0.2545, 0.2505, 0.2521])
        elif split == 'i':
            normalize = transforms.Normalize(mean=[0.4688, 0.4593, 0.4328], std=[0.2508, 0.2463, 0.2508])
        elif split == 'p':
            normalize = transforms.Normalize(mean=[0.4634, 0.4557, 0.4319], std=[0.2355, 0.2330, 0.2397])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if add_split is None or add_split in ['train', 'label', 'unlabel'] or add_split.startswith('train'):
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        dataset = CLEF_utils.CLEF_dataset(root=data_root, split=split, \
                                              transform=transform, add_split=add_split,
                                              ratio=ratio, resample=resample)
        loader = torch.utils.data.DataLoader(dataset, \
                                             batch_size=batch_size, shuffle=shuffle, \
                                             num_workers=4 * torch.cuda.device_count(), pin_memory=True)

    else:
        raise ('The selected dataset is not yet finished.')

    print ('[DATA LOADING] Loading from %s-%s-%s finish. Number of images: %d, Number of batches: %d' \
           %(dataset_name, split, add_split, len(loader.dataset), len(loader)))

    return loader


if __name__ == '__main__':
    pass