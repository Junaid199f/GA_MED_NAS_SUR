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
  #Setting the paramters for the GA-NAS algorithm
  remaining_datasets = ['organcmnist', 'organsmnist']
  datasets = ['octmnist','pneumoniamnist','breastmnist','bloodmnist','tissuemnist','organcmnist','organsmnist']
  for dataset in datasets:
    ga = GA(30,40,0.9,0.6,32,10,3,1,1024,8,16,0.3,True,True,False,3,5,'MEDMNIST',dataset)
    #Running the algorithm
    ga.evolve()
  # indv = [0.09, 0, 0.88, 0, 0.8, 0, 0.56, 0, 0.49, 1, 0.29, 0, 0.05, 0, 0.68, 2, 0.43, 0, 0.82, 0, 0.88, 0, 0.62, 0, 0.13, 0, 0.52, 1, 0.64, 3, 0.96, 2]
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, True, True, False, 3, 5, 'MEDMNIST', 'dermamnist')
  # ga.evaluate_single_model(indv)
  #indv = [0.2, 0, 0.25, 0, 0.12, 0, 0.72, 1, 0.07, 1, 0.55, 2, 0.87, 0, 0.02, 0, 0.2, 0, 0.93, 0, 0.86, 1, 0.77, 1, 0.7, 0, 0.66, 1, 0.48, 0, 0.59, 2]
  #ga.evaluate_single_model(indv)
  #Running the algorithm
  #ga.evolve()
  #
  #
  # ga = GA(20,40,0.9,0.6,32,10,3,1,1024,6,16,0.3,False,False,False,3,5,'MEDMNIST','bloodmnist')
  # ga.evolve()
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'pathmnist')
  # indv = [0.59, 0, 0.34, 0, 0.17, 0, 0.45, 1, 0.24, 0, 0.07, 0, 0.26, 1, 0.44, 0, 0.88, 0, 0.01, 0, 0.29, 0, 0.3, 0, 0.01, 1, 0.98, 0, 0.35, 3, 0.2, 1]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'chestmnist')
  # indv = [0.04, 0, 0.2, 0, 0.29, 1, 0.85, 0, 0.42, 1, 0.8, 0, 0.11, 1, 0.35, 0, 0.26, 0, 0.19, 0, 0.54, 1, 0.63, 0, 0.6, 2, 0.17, 0, 0.29, 2, 0.83, 1]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'dermamnist')
  # indv = [0.01, 0, 0.94, 0, 0.19, 0, 0.28, 0, 0.54, 0, 0.14, 1, 0.92, 1, 0.4, 1, 0.43, 0, 0.95, 0, 0.72, 0, 0.81, 1, 0.28, 2, 0.52, 2, 0.8, 3, 0.68, 3]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'octmnist')
  # indv = [0.44, 0, 0.48, 0, 0.38, 1, 0.42, 1, 0.17, 0, 0.04, 1, 0.81, 2, 0.5, 1, 0.75, 0, 0.21, 0, 0.23, 0, 0.69, 1, 0.42, 2, 0.62, 0, 0.24, 3, 0.68, 2]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'pneumoniamnist')
  # indv = [0.69, 0, 0.21, 0, 0.51, 0, 0.73, 1, 0.04, 2, 0.15, 2, 0.95, 1, 0.83, 1, 0.85, 0, 0.28, 0, 0.65, 1, 0.76, 0, 0.04, 2, 0.61, 2, 0.28, 1, 0.44, 1]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'retinamnist')
  # indv = [0.7, 0, 0.89, 0, 0.15, 1, 0.92, 0, 0.98, 2, 0.17, 1, 0.1, 2, 0.78, 1, 0.38, 0, 0.36, 0, 0.26, 1, 0.51, 0, 0.71, 2, 0.39, 1, 0.85, 2, 0.09, 1]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'breastmnist')
  # indv = [0.28, 0, 0.02, 0, 0.89, 0, 0.44, 1, 0.46, 1, 0.66, 0, 0.5, 3, 0.31, 2, 0.55, 0, 0.76, 0, 0.23, 0, 0.63, 1, 0.73, 0, 0.23, 1, 0.0, 2, 0.34, 3]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'bloodmnist')
  # indv = [0.67, 0, 0.92, 0, 0.88, 0, 0.16, 0, 0.22, 2, 0.2, 0, 0.59, 2, 0.45, 0, 0.21, 0, 0.49, 0, 0.9, 0, 0.33, 0, 0.9, 0, 0.52, 0, 0.87, 2, 0.96, 2]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'tissuemnist')
  # indv = [0.73, 0, 0.75, 0, 0.18, 0, 0.74, 1, 0.96, 1, 0.95, 0, 0.9, 0, 0.84, 2, 0.53, 0, 0.32, 0, 0.49, 0, 0.47, 0, 0.3, 2, 0.52, 1, 0.05, 2, 0.47, 0]
  # ga.evaluate_single_model(indv)
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 1024, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'organamnist')
  # indv =  [0.52, 0, 0.57, 0, 0.22, 0, 0.53, 1, 0.71, 0, 0.25, 0, 0.68, 0, 0.45, 3, 0.66, 0, 0.55, 0, 0.49, 1, 0.74, 0, 0.57, 0, 0.55, 1, 0.56, 0, 0.38, 1]
  # ga.evaluate_single_model(indv)
  # # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 512, 14, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'bloodmnist')
  # # indv = [0.19, 0, 0.4, 0, 0.44, 0, 0.37, 0, 0.45, 1, 0.71, 1, 0.06, 0, 0.77, 3, 0.89, 0, 0.81, 0, 0.27, 1, 0.09, 0, 0.82, 2, 0.17, 0, 0.09, 0, 0.07, 3]
  # # ga.evaluate_single_model(indv)
  #
  #
  # ga = GA(20, 40, 0.9, 0.6, 32, 10, 3, 1, 512, 8, 16, 0.3, False, False, False, 3, 5, 'MEDMNIST', 'organamnist')
  # indv = [0.21, 0, 0.11, 0, 0.05, 0, 0.21, 0, 0.6, 1, 0.81, 1, 0.53, 0, 0.23, 3, 0.55, 0, 0.42, 0, 0.42, 1, 0.98, 0, 0.79, 1,
  #  0.71, 1, 0.22, 3, 0.97, 1]
  # ga.evaluate_single_model(indv)