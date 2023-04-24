import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import math
import transforms as eeea_transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, datasets
from augmentations import Augmentation
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import medmnist
from medmnist import INFO, Evaluator
import utils
from pytorch_dataloader import MHIST, GasHisSDB


class Dataset:
  def __init__(self):
    self.augmentation = Augmentation()
    self.transforms = self.augmentation.get_augmentation()
  def get_mhist(self,batch_size,num_workers = 2):
      MHIST_path = os.path.join(os.getcwd(), 'DCPHB','images','images')
      MHIST_annoation_path = os.path.join(os.getcwd(),'DCPHB','annotations.csv')
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((198.2438, 166.4309, 188.5556), (41.3462, 58.2809, 47.7798)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((198.2438, 166.4309, 188.5556), (41.3462, 58.2809, 47.7798)),
      ])

      dataset = MHIST(MHIST_path, MHIST_annoation_path , transform_train)
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0,drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0,drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0,drop_last=True)
      classes = ['SSA', 'HP']

      return dataloader_train, dataloader_test, dataloader_val, classes
  def get_gashisdb(self,batch_size,num_workers=2):
      # percentage of training set to use as validation
      valid_size = 0.2

      # convert data to a normalized torch.FloatTensor
      # convert data to a normalized torch.FloatTensor
      transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((227.9496, 190.6923, 221.5993), (22.5608, 41.5429, 31.3247)),
      ])

      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((227.9496, 190.6923, 221.5993), (22.5608, 41.5429, 31.3247)),
      ])


      GasHisSDB_path = os.path.join(os.getcwd(),'GasHisSDB','GasHisSDB','160')
      # path=r"C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\GasHisSDB\GasHisSDB\160"
      # convert data to a normalized torch.FloatTensor
      dataset = GasHisSDB(GasHisSDB_path, transform_train)
      train_size = int(0.8 * len(dataset))
      test_size = len(dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
      train_size = int(0.8 * train_size)
      val_size = len(train_dataset) - train_size
      train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
      # dataset = MHIST(r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\images\images',r'C:\Users\omega\OneDrive\Documents\JUNAID CODES\Medical Datasets\DCPHB\annotations.csv')
      # classes = os.listdir(path) # list of subdirectories and files
      dataloader_train = DataLoader(train_dataset, batch_size=4,
                                    shuffle=True, num_workers=0,drop_last=True)
      dataloader_test = DataLoader(test_dataset, batch_size=4,
                                   shuffle=True, num_workers=0,drop_last=True)
      dataloader_val = DataLoader(valid_dataset, batch_size=4,
                                  shuffle=True, num_workers=0,drop_last=True)
      classes = ['Abnormal','Normal']


      return dataloader_train,dataloader_test,dataloader_val,classes

  def get_dataset_medmnist(self,dataset_name,batch_size):

      as_rgb =True
      shape_transform = False
      info = INFO[dataset_name]
      task = info['task']
      n_channels = 3 if as_rgb else info['n_channels']
      n_classes = len(info['label'])

      DataClass = getattr(medmnist, info['python_class'])


      print('==> Preparing data...')

      train_transform = utils.Transform3D(mul='random') if shape_transform else utils.Transform3D()
      eval_transform = utils.Transform3D(mul='0.5') if shape_transform else utils.Transform3D()

      train_dataset = DataClass(split='train', transform=train_transform, download='True', as_rgb=as_rgb)
      train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download='True', as_rgb=as_rgb)
      val_dataset = DataClass(split='val', transform=eval_transform, download='True', as_rgb=as_rgb)
      test_dataset = DataClass(split='test', transform=eval_transform, download='True', as_rgb=as_rgb)

      train_loader = data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
      train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                             batch_size=batch_size,
                                             shuffle=False)
      val_loader = data.DataLoader(dataset=val_dataset,
                                   batch_size=batch_size,
                                   shuffle=False)
      test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)

      return train_loader,val_loader,test_loader

  def get_dataset_fashionmnist(self, batch_size, num_workers=8):
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
        # train_transform, valid_transform = eeea_transforms._data_transforms_cifar10_search()
        # choose the training and test datasets
        train_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform_train)
        test_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform_test)

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
        return train_loader, valid_loader, test_loader
  def get_dataset_imagenet(self,batch_size,num_workers=8):
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
    train_data = datasets.ImageNet('data', train=True,
                                  download=True, transform=transform_train)
    test_data = datasets.ImageNet('data', train=False,
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
    return train_loader,valid_loader,test_loader
  def get_dataset_cifar100(self,batch_size,num_workers=8):
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
    train_data = datasets.CIFAR100('data', train=True,
                                  download=True, transform=transform_train)
    test_data = datasets.CIFAR100('data', train=False,
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
    return train_loader,valid_loader,test_loader

  def get_dataset(self,batch_size,num_workers=8):
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