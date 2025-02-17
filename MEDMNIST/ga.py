import json
import logging
import random
import os
import numpy as np
import random
import torch
import torchvision
import utils
import json
import time
import csv
import hashlib
from thop import profile
import torch.autograd.profiler as profiler
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from numpy import savetxt
from datetime import datetime
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
from model import NetworkCIFAR, Network, PyramidNetworkCIFAR
from utils import decode_cell, decode_operations
from optimizer import Optimizer
from Surrogate import Surrogate

class GA(Optimizer):
  def __init__(self,population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,retrain,resume_train,cutout,multigpu_num,grad_clip,dataset,medmnist_dataset,type_crossover):
    super().__init__(population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,retrain,resume_train,cutout,multigpu_num,grad_clip,dataset,medmnist_dataset,type_crossover)
  def crossover(self,individual1,individual2,prob_rate):
    offspring1 = []
    offspring2 = []
    gen_prob = random.uniform(0, 0.99)
    if gen_prob<=prob_rate:
      print("In crossover")
      while(1):
        crossover_rate = random.randint(0,len(individual1)-1)
        if crossover_rate>0:
          print("yes")
          offspring1 = individual1[:crossover_rate] + individual2[crossover_rate:]
          offspring2 = individual2[:crossover_rate] + individual1[crossover_rate:]
          return offspring1,offspring2
    else:
      return individual1,individual2


    gen_prob = random.uniform(0,0.99)
    if gen_prob<=mutation_prob:
      while(1):
        number = (random.randint(0,len(individual)-1))
        if  number%2 == 0:
          mutate_op = round(random.uniform(0,99),2)
          individual[number] = mutate_op
          return individual
    else:
      return individual

  def mutate(self, individual, mutation_prob):
    # Check if individual is a list and if it has more than one element
    if not isinstance(individual, list) or len(individual) < 2:
      raise ValueError("individual must be a list with at least two elements")

    # Check if mutation_prob is a number between 0 and 1
    if not (0 <= mutation_prob <= 1):
      raise ValueError("mutation_prob must be a number between 0 and 1")

    # Mutation for Genetic Algorithm
    gen_prob = random.uniform(0, 1)
    if gen_prob <= mutation_prob:
      while True:
        number = random.randint(0, len(individual) - 1)
        if number % 2 == 0:
          mutate_op = round(random.uniform(0, 0.99), 2)
          individual[number] = random.uniform(0, 0.99)
          return individual
    else:
      return individual
  def roullete_wheel_selection(self):
      population_fitness = sum([individual for individual in self.pop.fitness])
      chromosome_probabilities = [individual / population_fitness for individual in self.pop.fitness]
      # Making the probabilities for a minimization problem
      #chromosome_probabilities = 1 - np.array(chromosome_probabilities)
      rand_pick = np.random.choice(chromosome_probabilities)
      index_elem = chromosome_probabilities.index(rand_pick)
      return self.pop.individuals[index_elem]
  def binary_tournament_selection(self):
    while(1):
      indv1= random.randint(0,len(self.pop.individuals)-1)
      indv2= random.randint(0,len(self.pop.individuals)-1)
      if indv1 != indv2:
          if self.pop.fitness[indv1]<self.pop.fitness[indv2]:
            return self.pop.individuals[indv1]
          else:
            return self.pop.individuals[indv2]

  def enviroment_selection(self):
    while(1):
      indv1= random.randint(0,len(self.pop.individuals)-1)
      indv2= random.randint(0,len(self.pop.individuals)-1)
      #Euclidean distance between the individuals
      #if indv1 == indv2 :
        #indv1= random.randint(0,len(self.pop.individuals))
        #indv2= random.randint(0,len(self.pop.individuals))
      if indv1 != indv2 :
        return self.pop.individuals[indv1],self.pop.individuals[indv2]
  def evolve(self):
    # print(self.pop.indexes)
    # arr = [0.39728729056036727, 0, 0.6411476328852973, 0, 0.9084505613336283, 0, 0.5862298320167972, 1,
    #        0.5651331835892089,
    #        0, 0.6067150463828559, 0, 0.11323804584075525, 3, 98.39, 2, 0.004632023004602859, 0, 0.3469532018962277, 0,
    #        0.1763869865062233, 0, 0.6322642879195304, 1, 0.41462412372702373, 2, 0.7499604703991778, 1,
    #        0.47206694510999925,
    #        2, 0.44076274693822814, 1]
    # decode_operations(arr, self.pop.indexes)

    data_flag = self.medmnist_dataset
    output_root = './output'
    num_epochs = self.epochs
    gpu_ids = '0'
    batch_size = 64
    download = True
    run = 'model1'


    print("Parents_Trained_status: ",self.pop.parents_trained)
    print("Resume Training Status:", self.resume_train)
    #Get the current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    print("date and time =", dt_string)
    # If the parents individuals are not trained and the training is not resumed then train all the parents population
    if self.pop.parents_trained == False and self.resume_train == False :
      #Step 1: Get Individuals
      self.decoded_individuals = [NetworkCIFAR(self.n_channels,self.num_classes,self.layers,True,decode_cell(decode_operations(self.pop.individuals[i],self.pop.indexes)),self.dropout_rate,'FP32',False) for i in range(0,len(self.pop.individuals),1)]
      #Step 2: Train Individuals to Get Fitness Values
      for i,indv in enumerate(self.decoded_individuals):
        print("Parents Individual Number #", i)
        print("\n")
        hash_indv = hashlib.md5(str(self.pop.individuals[i]).encode("UTF-8")).hexdigest()
        loss = self.evaluator.train(indv,self.epochs,hash_indv,self.grad_clip,evaluation='valid',data_flag=data_flag, output_root=output_root, num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size, download=download, run=run)
        self.pop.fitness[i] = loss
        outfile = open("checkpoints/checkpoints.pkl","wb")
        pickle.dump(self.pop,outfile)
        outfile.close()
        # Saving Each parents population current index
        outfile_no = open("checkpoints/generations_pop.pkl", "wb")
        pickle.dump(i, outfile_no)
        outfile_no.close()

        #np.savetxt(os.path.join(os.path.join(os.getcwd(), 'logs'), "simples.csv"),
                   # self.pop.fitness,
                   # delimiter=", ",
                   # fmt='% s')
        pd.DataFrame(self.pop.fitness).to_csv(os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + dt_string + ".csv"), mode='w', encoding="utf-8", index=False)
      # print(self.pop.individuals)
      # print(self.pop.fitness)
      # datetime object containing current date and time
      #Setting parents_trained to true meaning all the parents population is trained
      self.pop.parents_trained=True
      outfile = open("checkpoints/checkpoints.pkl", "wb")
      pickle.dump(self.pop, outfile)
      outfile.close()
      #np.savetxt(os.path.join(os.path.join(os.getcwd(), "parents_logs_"+dt_string+".csv")), self.pop.fitness, delimiter=",")
      pd.DataFrame(self.pop.fitness).to_csv(os.path.join(os.path.join(os.getcwd(),'logs'), "parents_logs_"+dt_string+".csv"),mode='w',index=False)
      # with open(os.path.join(os.path.join(os.getcwd(), 'logs'), 'results_parents.json'),'w') as json_file:
      #   json.dump(self.pop.fitness, json_file)
      #Training Surrogate model
      train_data = np.asarray(self.pop.individuals)
      label = np.asarray(self.pop.fitness)
      self.surrogate.gbm_regressor(train_data, label)
    #check if the program stops during parents training, then resume the training from that index

      gen = 0
    elif self.pop.parents_trained == False and self.resume_train == True:
      # print("entered here")

      #Step 1: Get Individuals
      self.decoded_individuals = [NetworkCIFAR(self.n_channels,self.num_classes,self.layers,True,decode_cell(decode_operations(self.pop.individuals[i],self.pop.indexes)),self.dropout_rate,'FP32',False) for i in range(0,len(self.pop.individuals),1)]
      #Step 2: Train Individuals to Get Fitness Values
      outfile_no = open("checkpoints/generations_pop.pkl", "rb")
      gen_pop= pickle.load(outfile_no)
      outfile_no.close()
      print(gen_pop)
      print(len(self.decoded_individuals))
      for i in range(gen_pop,len(self.decoded_individuals)-1,1):
        print("Parents Individual Number #", i)
        print("\n")
        hash_indv = hashlib.md5(str(self.pop.individuals[i]).encode("UTF-8")).hexdigest()
        loss = self.evaluator.train(self.decoded_individuals[i], self.epochs, hash_indv,self.grad_clip)
        self.pop.fitness[i] = loss
        outfile = open("checkpoints/checkpoints.pkl", "wb")
        pickle.dump(self.pop, outfile)
        outfile.close()
        # Saving Each parents population current index
        outfile_no = open("checkpoints/generations_pop.pkl", "wb")
        pickle.dump(i, outfile_no)
        outfile_no.close()
        # datetime object containing current date and time
        pd.DataFrame(self.pop.fitness).to_csv(
          os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + str(now) + ".csv"), mode='w', index=False)
      self.pop.parents_trained = True
      outfile = open("checkpoints/checkpoints.pkl", "wb")
      pickle.dump(self.pop, outfile)
      outfile.close()
      # np.savetxt(os.path.join(os.path.join(os.getcwd(), "parents_logs_"+dt_string+".csv")), self.pop.fitness, delimiter=",")
      pd.DataFrame(self.pop.fitness).to_csv(
        os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + dt_string + ".csv"), mode='w', index=False)
      # with open(os.path.join(os.path.join(os.getcwd(), 'logs'), 'results_parents.json'),'w') as json_file:
      #   json.dump(self.pop.fitness, json_file)
    #If the parents are trained but the offspring population stops then the training starts from the previous generation
      train_data = np.asarray(self.pop.individuals)
      label = np.asarray(self.pop.fitness)
      self.surrogate.gbm_regressor(train_data, label)
      gen = 0
    elif self.pop.parents_trained==True and self.resume_train == True:
      isExist = os.path.exists("checkpoints/generation.pkl")
      if isExist:
        outfile_no = open("checkpoints/generation.pkl", "rb")
        gen = pickle.load(outfile_no)
        outfile_no.close()
      #If the parents are trained and the offspring population is not trained then this condition will run
      else:
        gen= 0

    #Train the surrogate
    #Step 3: Append the fitness values to Variable
    min_list_par = min(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
    fitn_par_g = self.pop.fitness[min_list_par]
    global gbest
    bestfitnesses = []
    gbest= fitn_par_g
    print("fitness",self.pop.fitness)
    #Step 4: For i to Number of Generations
    for i in range(gen, self.number_of_generations - 1, 1):
      logging.info("Generation Number # %s", i)
      print("\n")

      # Initialize offspring fitness and population
      self.offsprings_fitness = []
      self.offsprings_population = []

      # Set a limit of 80% of the population size for surrogate evaluation
      limit = int(self.population_size * 0.8)

      # First, evaluate 80% of the individuals using surrogate
      for j in range(self.population_size):
        # Step 5: Select individuals using binary tournament selection
        indv1 = self.binary_tournament_selection()
        indv2 = self.binary_tournament_selection()

        # Step 6: Perform crossover
        indv1, indv2 = self.crossover(indv1, indv2, self.crossover_prob)

        # Step 7: Perform mutation
        indv1 = self.mutate(indv1, self.mutation_prob)
        indv2 = self.mutate(indv2, self.mutation_prob)

        # Decode the individuals for evaluation
        decoded_indv1 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                     decode_cell(decode_operations(indv1, self.pop.indexes)),
                                     self.dropout_rate, 'FP32', False)
        decoded_indv2 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                     decode_cell(decode_operations(indv2, self.pop.indexes)),
                                     self.dropout_rate, 'FP32', False)

        # Create hash for individuals
        hash_indv1 = hashlib.md5(str(indv1).encode("UTF-8")).hexdigest()
        hash_indv2 = hashlib.md5(str(indv2).encode("UTF-8")).hexdigest()

        # Create checkpoint directories if they don't exist
        os.makedirs(os.path.join(os.getcwd(), 'checkpoints', hash_indv1), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'checkpoints', hash_indv2), exist_ok=True)

        # Evaluate the individuals: surrogate for 80%, manual for the remaining 20%
        if j < limit:
          individual_1_surrogate = [np.asarray(indv1)]
          individual_2_surrogate = [np.asarray(indv2)]

          # Surrogate evaluation
          loss_indv1 = self.surrogate.predict(individual_1_surrogate)
          loss_indv2 = self.surrogate.predict(individual_2_surrogate)
          logging.info("Surrogate predicted: loss_indv1: %s, loss_indv2: %s", loss_indv1, loss_indv2)
        else:
          # Manual evaluation through training
          loss_indv1 = self.evaluator.train(decoded_indv1, self.epochs, hash_indv1, self.grad_clip,
                                            evaluation='valid', data_flag=data_flag, output_root=output_root,
                                            num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,
                                            download=download, run=run)
          loss_indv2 = self.evaluator.train(decoded_indv2, self.epochs, hash_indv2, self.grad_clip,
                                            evaluation='valid', data_flag=data_flag, output_root=output_root,
                                            num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,
                                            download=download, run=run)
          logging.info("Manual training results: loss_indv1: %s, loss_indv2: %s", loss_indv1, loss_indv2)

        # Step 9: Append the offspring fitness and population
        self.offsprings_population.append(indv1)
        self.offsprings_population.append(indv2)
        self.offsprings_fitness.append(loss_indv1)
        self.offsprings_fitness.append(loss_indv2)

      # After evaluating the population, update the best individuals
      max_list_par = max(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
      fitn_par = self.pop.fitness[max_list_par]
      min_list_off = min(range(len(self.offsprings_fitness)), key=self.offsprings_fitness.__getitem__)
      fitn_off = self.offsprings_fitness[min_list_off]

      if fitn_off < fitn_par:
        self.pop.individuals[max_list_par] = self.offsprings_population[min_list_off]
        self.pop.fitness[max_list_par] = self.offsprings_fitness[min_list_off]

      # Update global best fitness if new offspring is better
      if fitn_off <= gbest:
        gbest = fitn_off

      logging.info("Gbest: %s", gbest)
      bestfitnesses.append(gbest)

      # Save logs and checkpoints
      pd.DataFrame(bestfitnesses).to_csv(os.path.join(os.getcwd(), 'logs', "offsprings_logs_" + dt_string + ".csv"),
                                         mode='w', index=False)
      with open("checkpoints/checkpoints.pkl", "wb") as outfile:
        pickle.dump(self.pop, outfile)
      with open("checkpoints/generation.pkl", "wb") as outfile_no:
        pickle.dump(i, outfile_no)

    #Step 11: End For Loop
    #Step 12: Print Results
    min_list_par = min(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
    print("Best Individual is with fitness value:: " , self.pop.fitness[min_list_par])
    print("\n Individual is \n", self.pop.individuals[min_list_par])
    self.evaluate_single_model(self.pop.individuals[min_list_par])
  def evaluate_single_model(self,indv,is_medmnist=True,check_power_consumption=True):


    network = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True, decode_cell(decode_operations(indv, self.pop.indexes)), self.dropout_rate, 'FP32', False)

    ### Computing Metrics ####
    if is_medmnist == True:
      input_tensor = torch.randn(4, 3, 32, 32)  # Replace with your input size
    else:
      input_tensor = torch.randn(4, 3, 256, 256)  # Replace with your input size

    input_tensor = input_tensor.cuda()
    model = network.cuda()
    if check_power_consumption == True:
      # Run a few warm-up iterations
      for _ in range(5):
        with torch.no_grad():
          _ = model(input_tensor)

      # Measure the time and profile the inference
      with profiler.profile(use_cuda=True) as prof:
        for _ in range(100):  # Adjust the number of iterations as needed
          with torch.no_grad():
            output = model(input_tensor)

      # Calculate the total inference time
      total_time_seconds = prof.self_cpu_time_total / 1000.0  # in seconds

      # Assuming you have power consumption information (in watts) for your device
      power_consumption_watts = 5.0  # Replace with the actual power consumption of your device

      # Estimate energy consumption (in joules)
      energy_consumption_joules = total_time_seconds * power_consumption_watts

      # Convert energy consumption to megajoules
      energy_consumption_megajoules = energy_consumption_joules / 1e6

      print(f"Total inference time: {total_time_seconds} seconds")
      print(f"Estimated energy consumption: {energy_consumption_megajoules} megajoules")
    flops, params = profile(model, inputs=(input_tensor,))
    size_in_mb = utils.count_parameters_in_MB(model)
    print(f"FLOPs: {flops / 1e9} billion")
    print(f"Parameters: {params / 1e6} million")
    with torch.no_grad():
      start_time = time.time()
      model(input_tensor)
      end_time = time.time()

    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Latency: {latency:.2f} ms")
    macs = flops / 2  # Divide by 2 since one MAC involves one multiplication and one addition
    print(f"MACs: {macs / 1e9} billion")

    ##############################################

    hash_indv = hashlib.md5(str(indv).encode("UTF-8")).hexdigest()
    loss = self.evaluator.train(network, 500, hash_indv,self.grad_clip,"test",self.medmnist_dataset, './output', 500, '0', 64, False, 'model1')

    print("loss",loss)
