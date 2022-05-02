from torchvision import datasets, transforms
import numpy as np
import torch
from tiny_imagenet import TinyImageNet
from torch.utils.data.distributed import DistributedSampler


datasets_dict = {
    'mnist': datasets.MNIST, 
    'svhn': datasets.SVHN, 
    'cifar10': datasets.CIFAR10,
    'cifar10_binary': datasets.CIFAR10,
    'cifar10_binary_gs': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'tiny_imagenet':TinyImageNet 
}


shapes_dict = {
    'mnist': (60000, 1, 28, 28), 
    'svhn': (73257, 3, 32, 32), 
    'cifar10': (50000, 3, 32, 32),
    'cifar10_binary': (10000, 3, 32, 32), 
    'cifar10_binary_gs': (10000, 1, 32, 32),
    'uniform_noise': (1000, 1, 28, 28),
    'cifar100': (50000, 3, 32, 32),
    'tiny_imagenet': (100000, 3, 64, 64)
}


def get_loaders(dataset, n_ex, batch_size, train_set, shuffle, data_augm, multiGPU=False):
    # print("n_ex{}".format(n_ex))
    dir_ = 'data/'
    dataset_f = datasets_dict[dataset]
    data_augm_transforms = [transforms.RandomCrop(32, padding=4)]
    if dataset == 'tiny_imagenet':
        data_augm_transforms = [transforms.RandomCrop(64, padding=4)]
    if dataset not in ['mnist', 'svhn']:
        # 数字类型的数据集不进行水平翻转
        data_augm_transforms.append(transforms.RandomHorizontalFlip())
    transform_list = data_augm_transforms if data_augm else []
    transform = transforms.Compose(transform_list + [transforms.ToTensor()])

    if "binary" in dataset:
        cl1, cl2 = 4, 8                                                        # 对于cifar10 (4,8) 对应 deers 和 ships


    if train_set:
        if dataset == 'tiny_imagenet':
            data = dataset_f(dir_, train=True,
            transform=transform, download=True)
        elif dataset != 'svhn':
            data = dataset_f(dir_, train=True, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='train', transform=transform, download=True)
        # print("ex_pre{}".format(n_ex))
        n_ex = len(data) if n_ex == -1 else n_ex
        # print("ex{}".format(n_ex))
        
        # 生成二分类的数据集
        if "binary" in dataset:
            data.targets = np.array(data.target)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)
        
        if "_gs" in dataset:
            data.data = data.data.mean(3).astype(np.unit8)
        
        if dataset == "svhn":
            data.targets = data.lable

        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]

        if multiGPU:
            train_sampler = DistributedSampler(data)
            loader = torch.utils.data.DataLoader(dataset=data, sampler=train_sampler, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True)

    else:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=False, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='test', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex
        # print("每k轮测试的样本个数{}".format(n_ex))
        # assert(n_ex == 320)
    
        if 'binary' in dataset:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)  # to reduce memory consumption
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        if dataset == 'svhn':
            data.targets = data.labels

        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]
        print("len(data){}".format(len(data.data)))
        print("len(dataset){}".format(len(data)))
        print("len(samples){}".format(len(data.dataset.samples)))
        # print(data.data)
        if multiGPU:
            test_sampler = DistributedSampler(data)
            loader = torch.utils.data.DataLoader(dataset=data, sampler=test_sampler, batch_size=batch_size, shuffle=shuffle, pin_memory=False, drop_last=False)
        else:
            loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False, drop_last=False)


    return loader




