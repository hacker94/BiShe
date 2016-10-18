# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 23:41:26 2016

@author: WANG Shaoyang
"""

import random

class Population:
    
    ##
    # @usage => constructor
    # @para0 => int, population size
    # @para1 => bool, initializing
    ##
    def __init__(self, population_size, need_init = True):
        self.individuals = [] # store individuals of this population
        if need_init:
            for i in xrange(population_size):
                individual = Individual()
                individual.generate_individual()
                self.save_individual(individual)
    
    ##
    # @usage => get an individual by index
    # @para0 => int, individual index
    # @return => Individual, the individual
    ##
    def get_individual(self, index):
        return self.individuals[index]
        
    ##
    # @usage => get an iter of all individuals
    # @return => listiterator, the iter of all individuals
    ##
    def get_individuals(self):
        return iter(self.individuals)
        
    ##
    # @usage => save an individual on index
    # @para0 => int, index
    # @para0 => Individual, individual 
    ##
    def save_individual(self, individual):
        self.individuals.append(individual)

    ##
    # @usage => get the size of this population
    # @return => int, size
    ##        
    def size(self):
        return len(self.individuals)
        
    ##
    # @usage => get the fittest individual in this population
    # @return => the fittest individual
    ##
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
        
        for indiv in new_pop.get_individuals(): 
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
        
        
if __name__ == '__main__':
    FitnessCalc.set_solution('1111000000000000000000000000000000000000000000000000000000001111')
    my_pop = Population(10)
    generation_cnt = 0
    while my_pop.get_fittest().get_fitness() < FitnessCalc.get_max_fitness():
        generation_cnt += 1
        print 'Generation: %d  Fittest: %d' % (generation_cnt, my_pop.get_fittest().get_fitness())
        my_pop = Algorithm.evolve_population(my_pop)
        
    print 'Solution found!'
    print 'Generation: %d' % generation_cnt
    print 'Genes: %s' % my_pop.get_fittest()
        