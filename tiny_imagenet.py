from typing import Tuple, Any, Callable, Optional
"""
Modified from
https://github.com/Clockware/nn-tiny-imagenet-200/blob/master/nn-tiny-imagenet-200.py
CREDIT: https://github.com/Clockware
"""
import random

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.utils
import torchvision.datasets as dsets
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
import zipfile
import os
from urllib.request import urlretrieve
from shutil import copyfile
from PIL import Image
import numpy as np
from torchvision import transforms as T

class TinyImageNet(VisionDataset) :
    def __init__(self, root="data",
                 train=True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download=False) :
        super(TinyImageNet, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        # self.transform = T.Compose([
        #                         T.Resize(224),  #缩放图片（Image）,保持长宽比不变，最短边为224像素
        #                         T.CenterCrop(224), #从图片中间裁剪出224*224的图片
        #                         T.ToTensor(), #将图片Image转换成Tensor，归一化至【0,1】
        #                         T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  #标准化至【-1,1】，规定均值和方差
        #                     ])
        
        if root[-1] == "/" :
            root = root[:-1]
        
        self._ensure_dataset_loaded(root)
        
        if train :
            self.dataset = dsets.ImageFolder(root+'/tiny-imagenet-200/train', 
                                          transform=transform)
            # self.data, self.targets = self.dataset.samples
            self.data = self.dataset.samples[:][0]
            self.targets = self.dataset.targets
        else :
            self.dataset = dsets.ImageFolder(root+'/tiny-imagenet-200/val_fixed',
                                          transform=transform)
            # self.data, self.targets = self.dataset.samples
            self.data = self.dataset.samples[:][0]
            self.targets = self.dataset.targets
        if download:
            self._download_dataset("./data")
        # self.sum = 0
        
    def _download_dataset(self, path,
                          url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                          tar_name='tiny-imagenet-200.zip'):
        if not os.path.exists(path):
            os.mkdir(path)
            
        if os.path.exists(os.path.join(path, tar_name)):
            print("Files already downloaded and verified")
            return
        else :
            print("Downloading Files...")
            urlretrieve(url, os.path.join(path, tar_name))
    #         print (os.path.join(path, tar_name))

            print("Un-zip Files...")
            zip_ref = zipfile.ZipFile(os.path.join(path, tar_name), 'r')
            zip_ref.extractall(path=path)
            zip_ref.close()

    def _ensure_dataset_loaded(self, root):
        self._download_dataset(root)

        val_fixed_folder = root+"/tiny-imagenet-200/val_fixed"
        if os.path.exists(val_fixed_folder):
            return
        os.mkdir(val_fixed_folder)

        with open(root+"/tiny-imagenet-200/val/val_annotations.txt") as f:
            for line in f.readlines():
                fields = line.split()

                file_name = fields[0]
                clazz = fields[1]

                class_folder = root+ "/tiny-imagenet-200/val_fixed/" + clazz
                if not os.path.exists(class_folder):
                    os.mkdir(class_folder)

                original_image_path = root+ "/tiny-imagenet-200/val/images/" + file_name
                copied_image_path = class_folder + "/" + file_name

                copyfile(original_image_path, copied_image_path)

    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        img_path, target = self.dataset.samples[index]
        
        # pil_img = Image.open(img_path)
        # array=np.asarray(pil_img)
        # img = torch.from_numpy(array)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print("{} {} {} {}".format(self.sum, index, img_path, img.shape))
        # self.sum += 1
        if img.shape[0] == 1:
            img = torch.cat((img, img, img))
        return img, target
