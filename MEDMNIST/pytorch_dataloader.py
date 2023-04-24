#This file implements the dataloaders for different medical datasets
#Written by Muhammad Junaid Ali for NAS-GA Framework

import os
import torch
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset



class MHIST(Dataset):
    def __init__(self,images_path,annotation_file,transforms=None):
        #print(images_path)
        self.annotation_file = pd.read_csv(annotation_file)
        images_names = self.annotation_file['Image Name'].values
        labels = self.annotation_file['Majority Vote Label']
        partitions = self.annotation_file['Partition']
        self.data = []
        for image_name,label,partition in zip(images_names,labels,partitions):
            self.data.append([os.path.join(images_path,image_name),label])
        self.class_map = {"HP": 0, "SSA": 1}
        self.img_dim = (224,224)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        #class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id
        #return img, class_id


class GasHisSDB(Dataset):
    def __init__(self, data_path,transforms=None):
        classes = os.listdir(data_path)
        self.transforms =transforms
        self.data = []
        for class_item in classes :
            for img_path in glob.glob(os.path.join(data_path , class_item) + "/*.png"):
                self.data.append([img_path, class_item])
        self.class_map = {"Abnormal": 0, "Normal": 1}
        self.img_dim = (256, 256)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id



