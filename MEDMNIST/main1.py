import random
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from ga import GA

if __name__ == '__main__':
  ga = GA(20,20,0.9,0.6,32,10,3,20,128,20,16,0.3)
  ga.evolve()