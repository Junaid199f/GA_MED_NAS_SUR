import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import math
import transforms as eeea_transforms
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, datasets
from augmentations import Augmentation
import torch
import torch.nn as nn
import torchvision

class Dataset:
  def __init__(self):
    self.augmentation = Augmentation()
    self.transforms = self.augmentation.get_augmentation()
  def get_dataset(self,batch_size,num_workers=2):
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #train_transform, valid_transform = eeea_transforms._data_transforms_cifar10_search()
    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform_train)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform_test)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    return train_loader,valid_loader,test_loader,classes