import math
import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import augment
import augmentations
from utils import create_param_choices
from dataset import Dataset
from evaluate import Evaluate
import genotype
import hashlib
import operations
import random
import operations_mapping
from operations import OPS
import lhsmdu

class Population():
  def __init__(self,block_size,population_size):
    self.element = []
    self.kernel_size = [3,5,7]
    self.pooling = [1,2]
    self.parents_trained=False
    self.normalization = [0,1,3]
    self.attention_layer = [1,2,3,4]
    self.pooling_filter = [2,4,6]
    self.dropout_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    self.intermediate_channels = [2 ** (i+1) for i in range(1,block_size+1,1)]
    self.block_size=block_size
    self.population_size = population_size
    self.networks_indvs = {}
    self.n_ops = len(operations_mapping.operations_mapping)
    self.indexes = self.setup_NAS(4,10)
    self.params_choices = create_param_choices(operations_mapping.primitives, self.indexes)
    self.fitness = np.zeros(self.population_size)
    self.individuals  = [ self.generate_individuals(self.block_size) for i in range(self.population_size)]

    #Creating directories for saving model logs and results of each individual
    for i,indv in enumerate(self.individuals):
      self.networks_indvs[hashlib.md5(str(indv).encode("UTF-8")).hexdigest()] = indv
      os.mkdir(os.path.join(os.path.join(os.getcwd(), 'checkpoints'), str(hashlib.md5(str(indv).encode("UTF-8")).hexdigest())))

  def setup_NAS(self,n_blocks, n_ops):
    n_var = int(4 * n_blocks * 2)
    ub = np.ones(n_var)
    h = 1
    for b in range(0, n_var//2, 4):
        ub[b] = n_ops
        ub[b + 1] = h
        ub[b + 2] = n_ops
        ub[b + 3] = h
        h += 1
    ub[n_var//2:] = ub[:n_var//2]
    return ub.astype(np.uint8)
  def generate_individuals(self,block_size):
    #sampler = qmc.LatinHypercube(d=1)
    #sample = sampler.random(n=1)
    self.individual = []
    for i in range(block_size):
     if i%2 == 0:
       #Uniform Selection
       #self.individual.append(round(random.uniform(0,0.99),2))
       #Latin Hypercube Sampling
       l_bounds = [0]
       u_bounds = [0.99]
       indv=lhsmdu.sample(1,1)
       indv = np.asarray(indv)
       self.individual.append(indv[0][0])
       #self.individual.append(round(random.uniform(0, 0.99), 2))
     else:
       self.individual.append(int(random.choice(self.params_choices[str(i)])))
    return self.individual

  def decode_individuals(self,pop):
    population = []
    for i,gen in enumerate(pop):
      network = {}
      for i,indv in enumerate(gen):
        #print(OPS.get(thisdict.get(indv)))
        if i%2 == 0:
          network[str(i)] = operations_mapping.get(math.floor((indv)*len(operations_mapping)))
          #print(operations_mapping.get(indv))
        # else:
        #   #print(int(random.choice(indexes)))
        #   network[str(i)] = int(random.choice(self.indexes))
      population.append(network)
    return population