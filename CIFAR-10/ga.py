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

    def save_population_checkpoint(self, index=None):
        checkpoint_dir = os.path.join(self.save, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, f'population_checkpoint_{index}.pkl' if index is not None else 'population_checkpoint.pkl')
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.pop, f)
        logging.info(f'Saved population checkpoint at {index if index is not None else "final"}')

    def train_initial_population(self):
        self.decoded_individuals = [NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True, decode_cell(
            decode_operations(self.pop.individuals[i], self.pop.indexes)), self.dropout_rate, 'FP32', False) for i
                                    in range(len(self.pop.individuals))]
        for i, indv in enumerate(self.decoded_individuals):
            logging.info("Parents Individual Number # %s", i)
            hash_indv = hashlib.md5(str(self.pop.individuals[i]).encode("UTF-8")).hexdigest()
            loss = self.evaluator.train(indv, self.epochs, hash_indv, self.grad_clip)
            logging.info("loss %s", loss)
            self.pop.fitness[i] = loss
            self.save_population_checkpoint(i)
        self.pop.parents_trained = True
        self.save_population_checkpoint()

        # Train the surrogate model with the initial population
        self.surrogate_individuals = deepcopy(self.pop.individuals)
        self.surrogate_individuals_fitness = deepcopy(self.pop.fitness)
        train_data = np.asarray(self.surrogate_individuals)
        label = np.asarray(self.surrogate_individuals_fitness)
        self.surrogate.gbm_regressor(train_data, label)

    def resume_training(self):
        self.decoded_individuals = [NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True, decode_cell(
            decode_operations(self.pop.individuals[i], self.pop.indexes)), self.dropout_rate, 'FP32', False) for i
                                    in range(len(self.pop.individuals))]
        gen_pop = self.load_population_checkpoint()
        for i in range(gen_pop, len(self.decoded_individuals)):
            logging.info("Parents Individual Number # %s", i)
            hash_indv = hashlib.md5(str(self.pop.individuals[i]).encode("UTF-8")).hexdigest()
            loss = self.evaluator.train(self.decoded_individuals[i], self.epochs, hash_indv, self.grad_clip)
            logging.info("loss %s", loss)
            self.pop.fitness[i] = loss
            self.save_population_checkpoint(i)
        self.pop.parents_trained = True
        self.save_population_checkpoint()

        # Train the surrogate model with the initial population
        self.surrogate_individuals = deepcopy(self.pop.individuals)
        self.surrogate_individuals_fitness = deepcopy(self.pop.fitness)
        train_data = np.asarray(self.surrogate_individuals)
        label = np.asarray(self.surrogate_individuals_fitness)
        self.surrogate.gbm_regressor(train_data, label)

        return 0

    def create_checkpoint_directory(self, hash_indv):
        checkpoint_dir = os.path.join(self.save, 'checkpoints', hash_indv)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        logging.info(f"Created checkpoint directory: {checkpoint_dir}")

    def evaluate_individual(self, network, hash_indv):
        """
        Evaluates an individual by training the model and returning the loss.
        """
        logging.info(f"Evaluating individual with hash: {hash_indv}")

        # Train the model using the evaluator and return the loss
        loss = self.evaluator.train(network, self.epochs, hash_indv, self.grad_clip)

        logging.info(f"Loss for individual {hash_indv}: {loss}")
        return loss

    def update_population(self):
        """
        Updates the population by selecting the best individuals from both
        the current population and newly generated offspring.
        """
        logging.info("Updating population...")

        # Combine current population with the offspring
        combined_population = self.pop.individuals + self.offsprings_population
        if not isinstance(self.pop.fitness, list):
            combined_fitness = self.pop.fitness.tolist() + self.offsprings_fitness
        else:
            combined_fitness = self.pop.fitness + self.offsprings_fitness

        # Sort the population based on fitness (assuming lower is better)
        # Convert all elements to scalars
        combined_fitness = [float(x) if isinstance(x, np.ndarray) else x for x in combined_fitness]

        # Now apply argsort
        sorted_indices = np.argsort(combined_fitness)
        #sorted_indices = np.argsort(combined_fitness)

        # Select the top individuals to form the new population
        self.pop.individuals = [combined_population[i] for i in sorted_indices[:self.population_size]]
        self.pop.fitness = [combined_fitness[i] for i in sorted_indices[:self.population_size]]

        logging.info(f"Updated population with {len(self.pop.individuals)} individuals.")

    def print_final_results(self):
        logging.info("Final results of the Genetic Algorithm:")

        # Get the best individual based on fitness
        best_index = np.argmin(self.pop.fitness)  # Assuming lower fitness is better
        best_individual = self.pop.individuals[best_index]
        best_fitness = self.pop.fitness[best_index]

        logging.info(f"Best Individual: {best_individual}")
        logging.info(f"Best Fitness Score: {best_fitness}")

        # Optionally, print the full population results
        logging.info("Final Population:")
        for i, (indv, fit) in enumerate(zip(self.pop.individuals, self.pop.fitness)):
            logging.info(f"Individual {i}: {indv}, Fitness: {fit}")

        print("Genetic Algorithm Evolution Completed.")
    def evolve(self):
        # Setup Logging Environment
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        logging.info("Parents_Trained_status: %s", self.pop.parents_trained)
        logging.info("Resume Training Status: %s", self.resume_train)
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

        if not self.pop.parents_trained and not self.resume_train:
            self.train_initial_population()
            gen = 0
        elif not self.pop.parents_trained and self.resume_train:
            gen = self.resume_training()
        elif self.pop.parents_trained and self.resume_train:
            gen = self.load_checkpoint()

        min_list_par = min(range(len(self.pop.fitness)), key=self.pop.fitness.__getitem__)
        fitn_par_g = self.pop.fitness[min_list_par]
        global gbest
        bestfitnesses = []
        gbest = fitn_par_g

        for i in range(gen, self.number_of_generations):
            logging.info("Generation Number # %s", i)
            self.offsprings_fitness = []
            self.offsprings_population = []

            for j in range(int(self.population_size * 0.8)):
                indv1 = self.binary_tournament_selection()
                indv2 = self.binary_tournament_selection()

                indv1, indv2 = self.crossover(indv1, indv2, self.crossover_prob, self.type_crossover)
                indv1 = self.mutate(indv1, self.mutation_prob)
                indv2 = self.mutate(indv2, self.mutation_prob)

                individual_1_surrogate = np.asarray(indv1).reshape(1, -1)
                individual_2_surrogate = np.asarray(indv2).reshape(1, -1)
                loss_indv1 = self.surrogate.predict(individual_1_surrogate)
                loss_indv2 = self.surrogate.predict(individual_2_surrogate)

                self.offsprings_population.append(indv1)
                self.offsprings_population.append(indv2)
                self.offsprings_fitness.append(loss_indv1)
                self.offsprings_fitness.append(loss_indv2)

            for j in range(int(self.population_size * 0.2)):
                indv1 = self.binary_tournament_selection()
                indv2 = self.binary_tournament_selection()

                indv1, indv2 = self.crossover(indv1, indv2, self.crossover_prob, self.type_crossover)
                indv1 = self.mutate(indv1, self.mutation_prob)
                indv2 = self.mutate(indv2, self.mutation_prob)

                decoded_indv1 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                             decode_cell(decode_operations(indv1, self.pop.indexes)), self.dropout_rate,
                                             'FP32', False)
                decoded_indv2 = NetworkCIFAR(self.n_channels, self.num_classes, self.layers, True,
                                             decode_cell(decode_operations(indv2, self.pop.indexes)), self.dropout_rate,
                                             'FP32', False)

                hash_indv1 = hashlib.md5(str(indv1).encode("UTF-8")).hexdigest()
                hash_indv2 = hashlib.md5(str(indv2).encode("UTF-8")).hexdigest()

                self.create_checkpoint_directory(hash_indv1)
                self.create_checkpoint_directory(hash_indv2)

                loss_indv1 = self.evaluate_individual(decoded_indv1, hash_indv1)
                loss_indv2 = self.evaluate_individual(decoded_indv2, hash_indv2)

                self.offsprings_population.append(indv1)
                self.offsprings_population.append(indv2)
                self.offsprings_fitness.append(loss_indv1)
                self.offsprings_fitness.append(loss_indv2)

            self.update_population()
            if not isinstance(self.surrogate_individuals_fitness, list):
                self.surrogate_individuals_fitness = self.surrogate_individuals_fitness.tolist()
            self.surrogate_individuals.extend(self.offsprings_population)
            self.surrogate_individuals_fitness.extend(self.offsprings_fitness)
            train_data = np.asarray(self.surrogate_individuals)
            self.surrogate_individuals_fitness = [
                float(x) if isinstance(x, (int, float)) else float(np.array(x).flatten()[0]) for x in
                self.surrogate_individuals_fitness]

            # Now, you can safely convert it into a NumPy array
            label = np.asarray(self.surrogate_individuals_fitness)
            self.surrogate.gbm_regressor(train_data, label)

            self.save_checkpoint(i)

        self.print_final_results()

    def save_checkpoint(self, index):
        """
        Saves the current state of the population, including individuals and fitness scores.
        """
        checkpoint_dir = os.path.join(self.save, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{index}.pkl')
        checkpoint_data = {
            'individuals': self.pop.individuals,
            'fitness': self.pop.fitness
        }

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        logging.info(f"Saved checkpoint at generation {index}")
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
