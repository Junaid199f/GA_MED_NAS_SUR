# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:48:33 2022

@author: Muhammad Junaid Ali (IRMAS Lab, University Haute Alsace)
"""


import os
import hashlib
import logging
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

import utils
from model import NetworkCIFAR
from optimizer import Optimizer
from utils import decode_cell, decode_operations


class GA(Optimizer):
    def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
                 in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
                 multigpu_num, grad_clip, type_crossover):
        super().__init__(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size,
                         num_classes, in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,
                         resume_train, cutout, multigpu_num, grad_clip, type_crossover)

    def crossover(self, individual1, individual2, prob_rate, type="one-point"):
        #Crossover for Genetic Algorithm
        offspring1 = []
        offspring2 = []
        if type == "one-point":
            gen_prob = random.uniform(0, 1)
            if gen_prob <= prob_rate:
                while (1):
                    crossover_rate = random.randint(0, len(individual1) - 1)
                    if crossover_rate > 0:
                        offspring1 = individual1[:crossover_rate] + individual2[crossover_rate:]
                        offspring2 = individual2[:crossover_rate] + individual1[crossover_rate:]
                        return offspring1, offspring2
            else:
                return individual1, individual2
        elif type == 'two-point':
            gen_prob = random.uniform(0, 1)
            if gen_prob <= prob_rate:
                # print("In crossover")
                # print(gen_prob)
                # print(prob_rate)
                while (1):
                    # print("yes")
                    crossover_rate_1 = random.randint(0, len(individual1) - 1)
                    crossover_rate_2 = random.randint(0, len(individual1) - 1)
                    if crossover_rate_1 > 0 and crossover_rate_2 < len(
                            individual1) - 1 and crossover_rate_1 < crossover_rate_2:
                        print("yes")
                        offspring1 = individual1[:crossover_rate_1] + individual2[
                                                                      crossover_rate_1:crossover_rate_2] + individual1[
                                                                                                           crossover_rate_2:]
                        offspring2 = individual2[:crossover_rate_1] + individual1[
                                                                      crossover_rate_1:crossover_rate_2] + individual2[
                                                                                                           crossover_rate_2:]
                        return offspring1, offspring2
            else:
                return individual1, individual2
        elif type == 'uniform':
            offspring1 = []
            offspring2 = []
            for i in range(0, len(individual1)):
                if i % 2 == 0:
                    if random.uniform(0, 1) < prob_rate:
                        offspring1.append(individual2[i])
                        offspring2.append(individual1[i])
                    else:
                        offspring2.append(individual2[i])
                        offspring1.append(individual1[i])
                else:
                    offspring2.append(individual2[i])
                    offspring1.append(individual1[i])
            return offspring1, offspring2
        else:
            print("Wrong cross-over type. Kindly change the crossover type")
            return individual1, individual2

    def mutate(self, individual, mutation_prob):
        # Mutation for Genetic Algorithm
        gen_prob = random.uniform(0, 1)
        if gen_prob <= mutation_prob:
            while (1):
                number = (random.randint(0, len(individual) - 1))
                if number % 2 == 0:
                    mutate_op = round(random.uniform(0, 0.99), 2)
                    individual[number] = mutate_op
                    return individual
        else:
            return individual

    def roullete_wheel_selection(self):
        # Roullete Wheel Selection
        population_fitness = sum([individual for individual in self.pop.fitness])
        chromosome_probabilities = [individual / population_fitness for individual in self.pop.fitness]
        # Making the probabilities for a minimization problem
        # chromosome_probabilities = 1 - np.array(chromosome_probabilities)
        rand_pick = np.random.choice(chromosome_probabilities)
        index_elem = chromosome_probabilities.index(rand_pick)
        return self.pop.individuals[index_elem]

    def binary_tournament_selection(self):
        # Tournament Selection
        while (1):
            indv1 = random.randint(0, len(self.pop.individuals) - 1)
            indv2 = random.randint(0, len(self.pop.individuals) - 1)
            if indv1 != indv2:
                if self.pop.fitness[indv1] < self.pop.fitness[indv2]:
                    return self.pop.individuals[indv1]
                else:
                    return self.pop.individuals[indv2]

    def enviroment_selection(self):
        # Enviroment Selection
        while (1):
            indv1 = random.randint(0, len(self.pop.individuals) - 1)
            indv2 = random.randint(0, len(self.pop.individuals) - 1)
            if indv1 != indv2:
                return self.pop.individuals[indv1], self.pop.individuals[indv2]

    def evolve(self):
        # Setup Logging Enviroment #
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        logging.info("Parents_Trained_status: %s", self.pop.parents_trained)
        logging.info("Resume Training Status: %s", self.resume_train)
        # Get the current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        print("date and time =", dt_string)
        # If the parents individuals are not trained and the training is not resumed then train all the parents population
        if self.pop.parents_trained == False and self.resume_train == False:
            # Step 1: Get Individuals
            self.decoded_individuals = [NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True, decode_cell(
                decode_operations(self.pop.individuals[i], self.pop.indexes)), self.dropout_rate, 'FP32', False) for i
                                        in range(0, len(self.pop.individuals), 1)]
            # Step 2: Train Individuals to Get Fitness Values
            for i, indv in enumerate(self.decoded_individuals):
                logging.info("Parents Individual Number # %s", i)
                print("\n")
                hash_indv = hashlib.md5(str(self.pop.individuals[i]).encode("UTF-8")).hexdigest()
                loss = self.evaluator.train(indv, self.epochs, hash_indv, self.grad_clip)
                logging.info("loss %s", loss)
                self.pop.fitness[i] = loss
                outfile = open("checkpoints/checkpoints.pkl", "wb")
                pickle.dump(self.pop, outfile)
                outfile.close()
                # Saving Each parents population current index
                outfile_no = open("checkpoints/generations_pop.pkl", "wb")
                pickle.dump(i, outfile_no)
                outfile_no.close()

                pd.DataFrame(self.pop.fitness).to_csv(
                    os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + dt_string + ".csv"), mode='w',
                    encoding="utf-8", index=False)
            # Copying individuals to surrogate array for training
            self.surrogate_individuals = deepcopy(self.pop.individuals)
            self.surrogate_individuals_fitness = deepcopy(self.pop.fitness)
            # datetime object containing current date and time
            # Setting parents_trained to true meaning all the parents population is trained
            self.pop.parents_trained = True
            outfile = open("checkpoints/checkpoints.pkl", "wb")
            pickle.dump(self.pop, outfile)
            outfile.close()
            # np.savetxt(os.path.join(os.path.join(os.getcwd(), "parents_logs_"+dt_string+".csv")), self.pop.fitness, delimiter=",")
            pd.DataFrame(self.pop.fitness).to_csv(
                os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + dt_string + ".csv"), mode='w',
                index=False)
            # with open(os.path.join(os.path.join(os.getcwd(), 'logs'), 'results_parents.json'),'w') as json_file:
            #   json.dump(self.pop.fitness, json_file)
            # Training Surrogate model
            start = time.process_time()
            train_data = np.asarray(self.surrogate_individuals)
            label = np.asarray(self.surrogate_individuals_fitness)
            self.surrogate.gbm_regressor(train_data, label)
            logging.info("Time taken by surrogate to train the model %s", time.process_time() - start)

            # check if the program stops during parents training, then resume the training from that index

            gen = 0
        elif self.pop.parents_trained == False and self.resume_train == True:

            # Copying individuals to surrogate array for training

            self.surrogate_individuals = deepcopy(self.pop.individuals)
            self.surrogate_individuals_fitness = deepcopy(self.pop.fitness)
            # Step 1: Get Individuals
            self.decoded_individuals = [NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True, decode_cell(
                decode_operations(self.pop.individuals[i], self.pop.indexes)), self.dropout_rate, 'FP32', False) for i
                                        in range(0, len(self.pop.individuals), 1)]
            # Step 2: Train Individuals to Get Fitness Values
            outfile_no = open("checkpoints/generations_pop.pkl", "rb")
            gen_pop = pickle.load(outfile_no)
            outfile_no.close()
            print(gen_pop)
            print(len(self.decoded_individuals))
            for i in range(gen_pop, len(self.decoded_individuals) - 1, 1):
                logging.info("Parents Individual Number # %s", i)
                print("\n")
                hash_indv = hashlib.md5(str(self.pop.individuals[i]).encode("UTF-8")).hexdigest()
                loss = self.evaluator.train(self.decoded_individuals[i], self.epochs, hash_indv, self.grad_clip)
                logging.info("loss %s", loss)
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
                    os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + str(now) + ".csv"), mode='w',
                    index=False)
            self.pop.parents_trained = True
            outfile = open("checkpoints/checkpoints.pkl", "wb")
            pickle.dump(self.pop, outfile)
            outfile.close()
            pd.DataFrame(self.pop.fitness).to_csv(
                os.path.join(os.path.join(os.getcwd(), 'logs'), "parents_logs_" + dt_string + ".csv"), mode='w',
                index=False)
            #   json.dump(self.pop.fitness, json_file)
            # If the parents are trained but the offspring population stops then the training starts from the previous generation
            self.surrogate_individuals = self.pop.individuals
            self.surrogate_individuals_fitness = self.pop.fitness
            # Training Surrogate model
            start = time.process_time()
            train_data = np.asarray(self.surrogate_individuals)
            label = np.asarray(self.surrogate_individuals_fitness)
            self.surrogate.gbm_regressor(train_data, label)
            logging.info("Time taken by surrogate to train the model %s", time.process_time() - start)
            gen = 0
        elif self.pop.parents_trained == True and self.resume_train == True:
            self.surrogate_individuals = deepcopy(self.pop.individuals)
            self.surrogate_individuals_fitness = deepcopy(self.pop.fitness)
            isExist = os.path.exists("checkpoints/generation.pkl")
            if isExist:
                outfile_no = open("checkpoints/generation.pkl", "rb")
                gen = pickle.load(outfile_no)
                outfile_no.close()
            # If the parents are trained and the offspring population is not trained then this condition will run
            else:
                gen = 0

        # Train the surrogate
        # Step 3: Append the fitness values to Variable
        min_list_par = min(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
        fitn_par_g = self.pop.fitness[min_list_par]
        global gbest
        bestfitnesses = []
        gbest = fitn_par_g
        print("fitness", self.pop.fitness)
        # Running the evolution process
        for i in range(gen, self.number_of_generations - 1, 1):
            logging.info("Generation Number # %s", i)
            print("\n")
            self.offsprings_fitness = []
            self.offsprings_population = []
            # Go through all the population in each generations
            for j in range(self.population_size - int(0.2 * self.population_size)):
                # Step 5: Get Individuals and check with surrogate


                indv1 = self.binary_tournament_selection()
                indv2 = self.binary_tournament_selection()

                # Step 6: Do Crossover
                indv1, indv2 = self.crossover(indv1, indv2, self.crossover_prob, self.type_crossover)
                # Step 7: Do Mutation
                indv1 = self.mutate(indv1, self.mutation_prob)
                indv2 = self.mutate(indv2, self.mutation_prob)
                print(indv1)
                print(indv2)
                # Step 8: Get fitness values of two offsprings
                decoded_indv1 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                             decode_cell(decode_operations(indv1, self.pop.indexes)), self.dropout_rate,
                                             'FP32', False)
                decoded_indv2 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                             decode_cell(decode_operations(indv2, self.pop.indexes)), self.dropout_rate,
                                             'FP32', False)
                hash_indv1 = hashlib.md5(str(indv1).encode("UTF-8")).hexdigest()
                hash_indv2 = hashlib.md5(str(indv2).encode("UTF-8")).hexdigest()
                isExist = os.path.exists((os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                                       str(hashlib.md5(str(indv1).encode("UTF-8")).hexdigest()))))
                if not isExist:
                    os.mkdir(os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                          str(hashlib.md5(str(indv1).encode("UTF-8")).hexdigest())))
                isExist = os.path.exists((os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                                       str(hashlib.md5(str(indv2).encode("UTF-8")).hexdigest()))))
                if not isExist:
                    os.mkdir(os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                          str(hashlib.md5(str(indv2).encode("UTF-8")).hexdigest())))
                individual_1_surrogate = np.asarray(indv1)
                individual_1_surrogate = [indv1]
                individual_2_surrogate = np.asarray(indv2)
                individual_2_surrogate = [indv2]
                start = time.process_time()
                loss_indv1 = self.surrogate.predict(individual_1_surrogate)
                logging.info("Time taken by surrogate to predict the fitness of the model %s",
                             time.process_time() - start)
                loss_indv2 = self.surrogate.predict(individual_2_surrogate)
                logging.info("Loss for individual Number 1 predicted by surrogate %s", loss_indv1)
                logging.info("Loss for individual Number 2 predicted by surrogate %s", loss_indv2)
                # Step 9: Append Fitness Values
                self.offsprings_population.append(indv1)
                self.offsprings_population.append(indv2)
                self.offsprings_fitness.append(loss_indv1)
                self.offsprings_fitness.append(loss_indv2)
            # Retrain the surrogate on new data

            for j in range(self.population_size - int(0.2 * self.population_size), self.population_size):
                # Step 5: Get Individuals and check with surrogate
                # indv1,indv2 = self.enviroment_selection()
                indv1 = self.binary_tournament_selection()
                indv2 = self.binary_tournament_selection()

                # Step 6: Do Crossover
                indv1, indv2 = self.crossover(indv1, indv2, self.crossover_prob, self.type_crossover)
                # Step 7: Do Mutation
                indv1 = self.mutate(indv1, self.mutation_prob)
                indv2 = self.mutate(indv2, self.mutation_prob)
                print(indv1)
                print(indv2)
                # Step 8: Get fitness values of two offsprings
                decoded_indv1 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                             decode_cell(decode_operations(indv1, self.pop.indexes)), self.dropout_rate,
                                             'FP32', False)
                decoded_indv2 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                             decode_cell(decode_operations(indv2, self.pop.indexes)), self.dropout_rate,
                                             'FP32', False)
                hash_indv1 = hashlib.md5(str(indv1).encode("UTF-8")).hexdigest()
                hash_indv2 = hashlib.md5(str(indv2).encode("UTF-8")).hexdigest()
                isExist = os.path.exists((os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                                       str(hashlib.md5(str(indv1).encode("UTF-8")).hexdigest()))))
                if not isExist:
                    os.mkdir(os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                          str(hashlib.md5(str(indv1).encode("UTF-8")).hexdigest())))
                isExist = os.path.exists((os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                                       str(hashlib.md5(str(indv2).encode("UTF-8")).hexdigest()))))
                if not isExist:
                    os.mkdir(os.path.join(os.path.join(os.getcwd(), 'checkpoints'),
                                          str(hashlib.md5(str(indv2).encode("UTF-8")).hexdigest())))
                individual_1_surrogate = np.asarray(indv1)
                # individual_1_surrogate = [indv1]
                individual_2_surrogate = np.asarray(indv2)
                # individual_2_surrogate = [indv2]
                loss_indv1 = self.evaluator.train(decoded_indv1, self.epochs, hash_indv1, self.grad_clip)
                loss_indv2 = self.evaluator.train(decoded_indv2, self.epochs, hash_indv2, self.grad_clip)
                logging.info("Loss of individual in generations individual 1 %s", loss_indv1)
                logging.info("Loss of individual in generations individual 2 %s", loss_indv2)
                print(loss_indv1)
                print(loss_indv2)
                # self.surrogate_individuals.append = self.surrogate_individuals.append.tolist()
                self.surrogate_individuals.append(indv1)
                self.surrogate_individuals.append(indv2)
                self.surrogate_individuals_fitness = self.surrogate_individuals_fitness.tolist()
                self.surrogate_individuals_fitness.append(loss_indv1)
                self.surrogate_individuals_fitness.append(loss_indv2)
                self.surrogate_individuals_fitness = np.array(self.surrogate_individuals_fitness)

                # Step 9: Append Fitness Values
                self.offsprings_population.append(indv1)
                self.offsprings_population.append(indv2)
                self.offsprings_fitness.append(loss_indv1)
                self.offsprings_fitness.append(loss_indv2)

            # Retraining Surrogate model
            start = time.process_time()
            train_data = np.asarray(self.surrogate_individuals)
            label = np.asarray(self.surrogate_individuals_fitness)
            self.surrogate.gbm_regressor(train_data, label)
            logging.info("Time by surrogate to retrain %s", time.process_time() - start)
            # Step 10: Sort the fitness values to get top most value
            max_list_par = max(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
            fitn_par = self.pop.fitness[max_list_par]
            min_list_off = min(range(len(self.offsprings_fitness)), key=self.offsprings_fitness.__getitem__)
            fitn_off = self.offsprings_fitness[min_list_off]
            if fitn_off < fitn_par:
                self.pop.individuals[max_list_par] = self.offsprings_population[min_list_off]
                self.pop.fitness[max_list_par] = self.offsprings_fitness[min_list_off]
            # np.savetxt(os.path.join(os.path.join(os.getcwd(), "offsprings_logs_" + dt_string + ".csv")), self.pop.individuals,delimiter=",")
            if fitn_off <= gbest:
                gbest = fitn_off
            print("Gbest is ", gbest)
            print("fITNESS OFFSPRING IS", fitn_off)
            bestfitnesses.append(gbest)
            pd.DataFrame(bestfitnesses).to_csv(
                os.path.join(os.path.join(os.getcwd(), 'logs'), "offsprings_logs_" + dt_string + ".csv"), mode='w',
                index=False)
            # Saving Each offsprings population
            outfile = open("checkpoints/checkpoints.pkl", "wb")
            pickle.dump(self.pop, outfile)
            outfile.close()
            outfile_no = open("checkpoints/generation.pkl", "wb")
            pickle.dump(i, outfile_no)
            outfile_no.close()

        # Step 11: End For Loop
        # Step 12: Print Results
        min_list_par = min(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
        print("Best Individual is with fitness value:: ", self.pop.fitness[min_list_par])
        print("\n Individual is \n", self.pop.individuals[min_list_par])
        self.evaluate_single_model(self.pop.individuals[min_list_par])

    def evaluate_single_model(self, indv):
        #This function is to train the final model

        logging.info("Evaluating single model")
        network = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                               decode_cell(decode_operations(indv, self.pop.indexes)), self.dropout_rate, 'FP32', False)
        hash_indv = hashlib.md5(str(indv).encode("UTF-8")).hexdigest()
        loss = self.evaluator.train_final(network, 3500, hash_indv, self.grad_clip)
        print("loss", loss)

    def reload_training(self, indv):
        network = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                               decode_cell(decode_operations(indv, self.pop.indexes)), self.dropout_rate, 'FP32', False)
        hash_indv = hashlib.md5(str(indv).encode("UTF-8")).hexdigest()
        utils.load(network, os.path.join(os.getcwd(), 'checkpoint', 'ckpt.pth'))
        loss = self.evaluator.train(network, 8000, hash_indv, self.grad_clip)
        print("loss", loss)
