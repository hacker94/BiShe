# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 23:41:26 2016

@author: WANG Shaoyang
"""

import random
import csv
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
import time


class Population:

    def __init__(self, population_size, need_init=True):
        self.individuals = []  # store individuals of this population
        if need_init:
            for i in xrange(population_size):
                individual = Individual()
                individual.generate_individual()
                self.append_individual(individual)

    def get_individual(self, index):
        return self.individuals[index]

    def get_individuals(self):
        return self.individuals

    def append_individual(self, individual):
        self.individuals.append(individual)

    def size(self):
        return len(self.individuals)

    def get_fittest(self):
        if self.size() == 0:
            return None
        individuals = self.get_individuals()
        fittest = individuals[0]
        for individual in individuals[1:]:
            if individual.get_fitness() > fittest.get_fitness():
                fittest = individual
        return fittest


class Individual:

    default_genes_len = 64
    default_gene_range = 10000.0
    default_mutation_step = 10000.0

    def __init__(self):
        self.genes = np.zeros(Individual.default_genes_len).astype(np.float32)
        self.fitness = 0

    def generate_individual(self):
        self.genes = np.zeros(Individual.default_genes_len).astype(np.float32)
        for i in xrange(Individual.default_genes_len):
            self.genes[i] = random.uniform(-Individual.default_gene_range, Individual.default_gene_range)

    @staticmethod
    def set_default_genes_len(length):
        Individual.default_genes_len = length

    def get_gene(self, index):
        return self.genes[index]

    def set_gene(self, index, val):
        self.genes[index] = val
        self.fitness = 0

    def get_fitness(self):
        if self.fitness == 0:
            self.fitness = FitnessCalc.get_fitness(self)
        return self.fitness

    def get_genes(self):
        return self.genes


class Algorithm:

    uniform_rate = 0.85
    mutation_rate = 0.05
    tournament_size = 30
    b = 3
    elitism = True

    @staticmethod
    def evolve_population(pop, process_percentage):
        if not isinstance(pop, Population):
            raise TypeError('need a Population')

        new_pop = Population(pop.size(), False)

        if Algorithm.elitism:
            new_pop.append_individual(pop.get_fittest())

        for i in xrange(pop.size() - (1 if Algorithm.elitism else 0)):
            indiv1 = Algorithm.tournament_selection(pop)
            indiv2 = Algorithm.tournament_selection(pop)
            new_indiv = Algorithm.crossover(indiv1, indiv2)
            new_pop.append_individual(new_indiv)

        for indiv in new_pop.get_individuals()[(1 if Algorithm.elitism else 0):]:
            Algorithm.mutate(indiv, process_percentage)

        return new_pop

    @staticmethod
    def tournament_selection(pop):
        tournament = Population(Algorithm.tournament_size, False)
        for i in xrange(Algorithm.tournament_size):
            tournament.append_individual(pop.get_individual(random.randint(0, pop.size() - 1)))
        return tournament.get_fittest()

    @staticmethod
    def crossover(indiv1, indiv2):
        new_indiv = Individual()
        for i in xrange(Individual.default_genes_len):
            if random.random() < Algorithm.uniform_rate:
                new_indiv.set_gene(i, indiv1.get_gene(i))
            else:
                new_indiv.set_gene(i, indiv2.get_gene(i))
        return new_indiv


    @staticmethod
    def non_uniform_degree(t, y):
        r = random.uniform(0, 1)
        return y * (1 - r ** ((1 - t) ** Algorithm.b))


    @staticmethod
    def mutate(indiv, process_percentage):
        if not isinstance(indiv, Individual):
            raise TypeError('need an Individual')

        for i in xrange(Individual.default_genes_len):
            if random.random() < Algorithm.mutation_rate:
                gene = indiv.get_gene(i)

                ra = random.randint(0, 1)  # a random bit
                if ra == 0:
                    gene += Algorithm.non_uniform_degree(process_percentage, Individual.default_mutation_step)
                else:
                    gene -= Algorithm.non_uniform_degree(process_percentage, Individual.default_mutation_step)

                indiv.set_gene(i, gene)


class FitnessCalc:

    data_X = None
    data_y = None
    X_gpu = None
    theta_gpu = None
    rst_gpu = None
    
    mod = SourceModule("""
    __global__ void h(const int *X, const float *theta, float *rst, int n, int m) {
        int i = threadIdx.x + blockIdx.x * 1024;
        if (i >= m) {
            return;
        }
        int j = 0;
        float t = 0;
        for (j = 0; j < n; j++) {
            t += X[i * n + j] * theta[j];
        }
        rst[i] = 1 / (1 + exp(-t / n * 0.0001));
    }
    """)
    h = mod.get_function("h")
    
    @staticmethod
    def h_cpu(theta, x):
        return 1 / (1 + np.exp(-np.dot(x, theta) / theta.shape[0] * 0.0001))  # x can be a vector or a matrix

    @staticmethod
    def l(theta):
        X = FitnessCalc.data_X
        y = FitnessCalc.data_y
        m = X.shape[0]
        n = X.shape[1]
        
        if FitnessCalc.X_gpu == None:
            FitnessCalc.X_gpu = cuda.mem_alloc(X.nbytes)
            cuda.memcpy_htod(FitnessCalc.X_gpu, X)
        if FitnessCalc.theta_gpu == None:
            FitnessCalc.theta_gpu = cuda.mem_alloc(theta.nbytes)
        cuda.memcpy_htod(FitnessCalc.theta_gpu, theta)
        rst = np.empty(m, dtype=np.float32)
        if FitnessCalc.rst_gpu == None:
            FitnessCalc.rst_gpu = cuda.mem_alloc(rst.nbytes)
        THREAD_CNT = 1024
        FitnessCalc.h(FitnessCalc.X_gpu, FitnessCalc.theta_gpu, FitnessCalc.rst_gpu, np.int32(n), np.int32(m), block=(THREAD_CNT, 1, 1), grid=(m / THREAD_CNT + 1, 1))
        cuda.memcpy_dtoh(rst, FitnessCalc.rst_gpu)
        
        fit = 1.0
        for i in xrange(m):
            r = float(rst[i])
            if y[i] == 1:
                fit *= r + r
            else:
                fit *= (1 - r) * 2
        return fit

    @staticmethod
    def get_fitness(indiv):
        theta = indiv.get_genes()
        fitness = FitnessCalc.l(theta)
        return fitness


def read(path, price_file, sentiment_file):
    with open(path + price_file, 'rb') as price_csv, open(path + sentiment_file, 'rb') as sentiment_csv:
        price_reader = csv.reader(price_csv)
        sentiment_reader = csv.reader(sentiment_csv)

        trends = {}
        last_price = 0
        for row in price_reader:
            price = float(row[4])
            date = row[0]
            if last_price != 0:
                trends[date] = 1 if price > last_price else 0
            last_price = price
        m = len(trends.keys())  # total data

        sentiments = {}
        event_id = {}

        n = 0  # total characteristics
        for row in sentiment_reader:
            date = row[0]
            event = (int(row[1]), int(row[2]), int(row[3]))
            if event not in event_id:
                event_id[event] = n
                n += 1

            weight = int(row[4])
            if date not in sentiments:
                sentiments[date] = {}
            sentiments[date][event] = weight

    return trends, sentiments, event_id, m, n


def build_vectors(trends, sentiments, event_id, m, n):
    X_all = np.zeros((m, n + 1))
    y_all = np.zeros(m)

    dates = sorted(trends.keys())
    for i in xrange(0, m):
        date = dates[i]
        for event in sentiments[date]:
            X_all[i, event_id[event]] = sentiments[date][event]
        X_all[i, n] = 1
        y_all[i] = trends[date]

    return X_all, y_all


def train(X, y):
    total_generation = 1000
    population_size = 100

    n = X.shape[1]

    Individual.set_default_genes_len(n)
    FitnessCalc.data_X = X
    FitnessCalc.data_y = y
    my_pop = Population(population_size)
    generation_cnt = 0
    t1 = time.clock()
    while generation_cnt < total_generation:
        generation_cnt += 1
        #print 'Generation: %d  Fittest: %f' % (generation_cnt, my_pop.get_fittest().get_fitness())
        my_pop = Algorithm.evolve_population(my_pop, float(generation_cnt) / total_generation)
    print 'Training time: %fs' % (time.clock() - t1)

    print 'Solution found!'
    print 'Generation: %d' % generation_cnt
    print 'Genes: %s' % my_pop.get_fittest().get_genes()
    return my_pop.get_fittest().get_genes()


def test(X, y, theta):
    total = y.shape[0]
    correct = 0
    for i in xrange(total):
        yh = FitnessCalc.h_cpu(theta, X[i])
        yh = 1 if yh > 0.5 else 0
        if yh == y[i]:
            correct += 1
    print 'Total: %d, Correct: %d, Accuracy: %f' % (total, correct, float(correct) / total)


if __name__ == '__main__':
    path = 'C:/Users/SyW/Desktop/BiShe/'
    #price_file = 'DAX-price-2-years.csv'
    #sentiment_file = 'DAX-sentiment-2-years.csv'
    price_file = 'GLD_price.csv'
    sentiment_file = 'GLD_Sentiment.csv'

    training_num = 300
    validation_num = training_num + 100

    # read trends and sentiments from file
    trends, sentiments, event_id, m, n = read(path, price_file, sentiment_file)

    # build vectors
    X_all, y_all = build_vectors(trends, sentiments, event_id, m, n)
    X_all = X_all.astype(np.int32)

    # work out theta
    X = X_all[0:training_num]
    y = y_all[0:training_num]
    '''Xt, yt = X, y
    for i in range(5):
        X = np.concatenate((X, Xt))
        y = np.concatenate((y, yt))'''
    theta = train(X, y)

    with open(path + 'out.txt', 'w') as fout:
        for t in theta:
            fout.write("%.9f\n" % t)

    # validation
    X_test = X_all[training_num:validation_num]
    y_test = y_all[training_num:validation_num]
    test(X_test, y_test, theta)

    # test
    final_test = True
    if final_test:
        X_test = X_all[validation_num:]
        y_test = y_all[validation_num:]
        test(X_test, y_test, theta)
