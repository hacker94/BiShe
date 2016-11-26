# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 23:41:26 2016

@author: WANG Shaoyang
"""

import random
import csv
import numpy as np
from scipy.optimize import leastsq

class Population:
    
    def __init__(self, population_size, need_init = True):
        self.individuals = [] # store individuals of this population
        if need_init:
            for i in xrange(population_size):
                individual = Individual()
                individual.generate_individual()
                self.save_individual(individual)
    
    def get_individual(self, index):
        return self.individuals[index]
        
    def get_individuals(self):
        return iter(self.individuals)
        
    def save_individual(self, individual):
        self.individuals.append(individual)
  
    def size(self):
        return len(self.individuals)

    def get_fittest(self):
        if self.size() == 0:
            return None
        individuals = self.get_individuals()
        fittest = individuals.next()
        for individual in individuals:
            if individual.get_fitness() > fittest.get_fitness():
                fittest = individual
        return fittest
    
    
class Individual:
    
    default_genes_len = 64

    def __init__(self):
        self.genes = []
        self.fitness = 0
        
    def generate_individual(self):
        self.genes = []
        for i in xrange(Individual.default_genes_len):
            self.genes.append(random.randint(0, 1))
    
    @staticmethod
    def set_default_genes_len(length):       
        Individual.default_genes_len = length
    
    def get_gene(self, index):
        return self.genes[index]
        
    def append_gene(self, val):
        self.genes.append(val)
        
    def set_gene(self, index, val):
        self.genes[index] = val
        self.fitness = 0        
        
    def get_fitness(self):
        if self.fitness == 0:
            self.fitness = FitnessCalc.get_fitness(self)
        return self.fitness
        
    def __str__(self):
        return ''.join([chr(x + ord('0')) for x in self.genes])
        

class Algorithm:
    
    uniform_rate = 0.5
    mutation_rate = 0.015
    tournament_size = 5
    elitism = True
    
    @staticmethod
    def evolve_population(pop):
        if not isinstance(pop, Population):
            raise TypeError('need a Population')
        
        new_pop = Population(pop.size(), False)
        
        if Algorithm.elitism:
            new_pop.save_individual(pop.get_fittest())
            
        for i in xrange(pop.size() - (1 if Algorithm.elitism else 0)):
            indiv1 = Algorithm.tournament_selection(pop)
            indiv2 = Algorithm.tournament_selection(pop)
            new_indiv = Algorithm.crossover(indiv1, indiv2)
            new_pop.save_individual(new_indiv)
        
        for indiv in new_pop.get_individuals()[(1 if Algorithm.elitism else 0):]: 
            Algorithm.mutate(indiv)
        
        return new_pop
        
    @staticmethod
    def tournament_selection(pop):
        tournament = Population(Algorithm.tournament_size, False)
        for i in xrange(Algorithm.tournament_size):
            tournament.save_individual(pop.get_individual(random.randint(0, pop.size() - 1)))
        return tournament.get_fittest()
        
    @staticmethod
    def crossover(indiv1, indiv2):
        new_indiv = Individual()
        for i in xrange(Individual.default_genes_len):
            if random.random() < Algorithm.uniform_rate:
                new_indiv.append_gene(indiv1.get_gene(i))
            else :
                new_indiv.append_gene(indiv2.get_gene(i))
        return new_indiv
    
    @staticmethod
    def mutate(indiv):
        if not isinstance(indiv, Individual):
            raise TypeError('need an Individual')
            
        for i in xrange(Individual.default_genes_len):
            if random.random() < Algorithm.mutation_rate:
                indiv.set_gene(i, random.randint(0, 1))
        
        
class FitnessCalc:
    
    solution = []
    
    @staticmethod
    def set_solution(new_solution):
        if isinstance(new_solution, list):
            FitnessCalc.solution = new_solution
        elif isinstance(new_solution, str):
            FitnessCalc.solution = [ord(x) - ord('0') for x in new_solution]
        else:
            raise TypeError('new solution need to be a list or str')
        
    @staticmethod
    def get_fitness(indiv):
        fitness = 0
        for i in xrange(Individual.default_genes_len):
            if indiv.get_gene(i) == FitnessCalc.solution[i]:
                fitness += 1
        return fitness
        
    @staticmethod
    def get_max_fitness():
        return len(FitnessCalc.solution)
        


def read():
    with open('DAX-price-2-years.csv', 'rb') as price_csv, open('DAX-sentiment-2-years.csv', 'rb') as sentiment_csv:
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
        m = len(trends.keys()) # total data
        
        sentiments = {}
        event_id = {}
        
        event_count = {}
        
        n = 0 # total characteristics
        for row in sentiment_reader:
            date = row[0]
            event = (int(row[1]), int(row[2]), int(row[3]))
            if not event in event_id:
                event_id[event] = n
                n += 1
                
                event_count[event] = 1
            else:
                event_count[event] += 1
                
            weight = int(row[4])
            if not date in sentiments:
                sentiments[date] = {}
            sentiments[date][event] = weight
            
    i = 0
    for event in event_count.keys():
        if event_count[event] < 200:
            event_id[event] = -1
            n -= 1
        else:
            event_id[event] = i
            i += 1
    
    return trends, sentiments, event_id, m, n
    
def build_vectors(trends, sentiments, event_id, m, n):
    X_all = np.zeros((m, n))
    y_all = np.zeros(m)
    
    dates = sorted(trends.keys())
    for i in xrange(0, m):
        date = dates[i]
        for event in sentiments[date]:
            if event_id[event] != -1:
                X_all[i, event_id[event]] = sentiments[date][event]
            #X_all[i, event_id[event]] = sentiments[date][event]
        y_all[i] = trends[date]
    
    return X_all, y_all
        
def h(theta, x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))  # x can be a vector or a matrix
    
def residuals(theta, X, y):
    return h(theta, X) - y
    
def train(X, y):
    theta = np.zeros(X.shape[1])
    theta = leastsq(residuals, theta, args = (X, y))[0]
    return theta
    
def test(X, y, theta):
    total = y.shape[0]
    correct = 0
    for i in xrange(total):
        yh = h(theta, X[i])
        yh = 1 if yh > 0.5 else 0
        if yh == y[i]:
            correct += 1
    print 'Total: %d, Correct: %d, Accuracy: %f' % (correct, total, float(correct) / total)
        
if __name__ == '__main__':
    '''
    FitnessCalc.set_solution('1111000000000000000000000000000000000000000000000000000000001111')
    my_pop = Population(50)
    generation_cnt = 0
    while my_pop.get_fittest().get_fitness() < FitnessCalc.get_max_fitness():
        generation_cnt += 1
        print 'Generation: %d  Fittest: %d' % (generation_cnt, my_pop.get_fittest().get_fitness())
        my_pop = Algorithm.evolve_population(my_pop)
        
    print 'Solution found!'
    print 'Generation: %d' % generation_cnt
    print 'Genes: %s' % my_pop.get_fittest()
    '''
    training_num = 400

    # read trends and sentiments from file
    trends, sentiments, event_id, m, n = read()
    
    # build vectors
    X_all, y_all = build_vectors(trends, sentiments, event_id, m, n)
        
    # work out theta
    X = X_all[0:training_num]
    y = y_all[0:training_num]
    theta = train(X, y)
    
    # test
    X_test = X_all[training_num:]
    y_test = y_all[training_num:]
    test(X_test, y_test, theta)
    