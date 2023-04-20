import os
import augment
import augmentations
from dataset import Dataset
from evaluate import Evaluate
import genotype
import operations
from operations_mapping import operations_mapping
from population import Population
import utils
import pickle
import Surrogate
import medmnist
from medmnist import INFO, Evaluator


class Optimizer:
  def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
               in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,resume_train,cutout,multigpu_num,grad_clip,dataset,medmnist_dataset):
    self.resume_train = resume_train
    if self.resume_train==True:
      z = open('checkpoints/checkpoints.pkl','rb')
      #self.pop =  Population(blocks_size, population_size)
      self.pop = pickle.load(z)
    else:
      self.pop = Population(blocks_size, population_size)
    self.population_size = population_size
    self.number_of_generations = number_of_generations
    self.medmnist_dataset = medmnist_dataset
    info = INFO[self.medmnist_dataset]
    task = info['task']
    n_channels = info['n_channels']
    self.num_classes = len(info['label'])

    self.crossover_prob = crossover_prob
    self.dataset = dataset
    #self.num_classes = num_classes
    if self.dataset == 'CIFAR-10':
      self.num_classes = 10
    elif self.dataset == 'CIFAR-100':
      self.num_classes = 100
    elif self.dataset == 'FASHIONMNIST':
      self.num_classes = 10
    elif self.dataset == 'IMAGENET':
      self.num_classes = 1000
    elif self.dataset == 'GASHISSDB':
      self.num_classes = 2
    elif self.dataset == 'MHIST':
      self.num_classes = 2
    self.epochs = epochs
    self.multigpu_num = multigpu_num
    self.gpu_devices = ','.join([str(id) for id in range(0,self.multigpu_num)])
    os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
    self.surrogate = Surrogate.Surrogate()
    self.cutcout = cutout
    self.mutation_prob = mutation_prob
    self.batch_size = batch_size
    self.grad_clip = grad_clip
    self.blocks_size = blocks_size

    self.in_channels = in_channels
    self.layers = layers
    self.dropout_rate = dropout_rate
    self.n_channels = n_channels
    self.intermediate_channels = [2 ** (i + 1) for i in range(1, blocks_size + 1, 1)]
    self.retrain = retrain
    self.fitness = []
    self.networks = []
    self.trained_individuals = False
    self.decoded_individuals = []
    self.offsprings_population = []
    self.offsprings_fitness = []
    self.evaluator = Evaluate(self.batch_size,self.dataset,self.medmnist_dataset)

    def evolve():
      pass

  def encode(self):
    return None

  def decode(self):
    self.decoded_individuals = self.pop.decode_individuals(self.pop.Individuals)
    # for i in range(self.population_size):
    # model = Net(pop.individual[0],self.in_channels,self.intermediate_channels,self.num_classes,self.block_size)

  def optimize(self):
    self.evolve()



 