from torchvision import datasets, transforms
import numpy as np
import torch


datasets_dict = {
    'mnist': datasets.MNIST, 
    'svhn': datasets.SVHN, 
    'cifar10': datasets.CIFAR10,
    'cifar10_binary': datasets.CIFAR10,
    'cifar10_binary_gs': datasets.CIFAR10    
}


shapes_dict = {
    'mnist': (60000, 1, 28, 28), 
    'svhn': (73257, 3, 32, 32), 
    'cifar10': (50000, 3, 32, 32),
    'cifar10_binary': (10000, 3, 32, 32), 
    'cifar10_binary_gs': (10000, 1, 32, 32),
    'uniform_noise': (1000, 1, 28, 28)
}


def get_loaders(dataset, n_ex, batch_size, train_set, shuffle, data_augm):
    dir_ = 'data/'
    dataset_f = datasets_dict[dataset]
    num_workers = 2
    data_augm_transforms = [transforms.RandomCrop(32, padding=4)]
    if dataset not in ['mnist', 'svhn']:
        # 数字类型的数据集不进行水平翻转
        data_augm_transforms.append(transforms.RandomHorizontalFlip())
    transform_list = data_augm_transforms if data_augm else []
    transform = transforms.Compose(transform_list + [transforms.ToTensor()])

    if "binary" in dataset:
        cl1, cl2 = 4, 8                                                        # 对于cifar10 (4,8) 对应 deers 和 ships


    if train_set:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=True, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='train', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex
        
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

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, drop_last=True)

    else:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=False, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='test', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex
    
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

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False, num_workers=num_workers, drop_last=False)


    return loader




