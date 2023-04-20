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
import logging
from ga import GA

if __name__ == '__main__':
  #Setting the parameters for the GA-NAS algorithm
  population_size = 6
  number_of_generations = 5
  crossover_prob = 0.7
  mutation_prob = 0.6
  blocks_size = 32
  num_classes = 10
  in_channels = 3
  epochs = 1
  batch_size = 256
  layers = 8
  n_channels =16
  dropout_rate = 0.3
  retrain = False
  resume_train = False
  cutout = False
  multigpu_num =  3
  grad_clip = 5
  type_crossover = "uniform" # one-point, two-point, uniform
  ga = GA(population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,retrain,resume_train,cutout,multigpu_num,grad_clip,type_crossover)
  #Running the algorithm
  ga.evolve()
  #indv = [0.76, 0, 0.08, 0, 0.69, 0, 0.15, 1, 0.57, 0, 0.62, 1, 0.1, 2, 0.45, 3, 0.11, 0, 0.42, 0, 0.58, 1, 0.45, 1, 0.75, 0, 0.37, 2, 0.38, 1, 0.5, 1]
  #ga.evaluate_single_model(indv) 
  #ga.reload_training(indv)
