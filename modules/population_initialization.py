#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) 
 
 PopulationInitializer abstract class and basic initializer that generates
 a population of random binary strings of a given length

"""
#######################################################################
import abc

import numpy as np
from numpy import random

#Base abstract class for population initialization methods, that generate a population for the genetic optimization process
class PopulationInitializer(metaclass=abc.ABCMeta):
    def __init__(self, population_size):
        self.population_size = population_size

    @abc.abstractmethod
    def generate_population(self):
        pass

    @abc.abstractmethod
    def generate_individual(self):
        pass

#Generate population of random binary strings of a given length
class BinaryPopulationInitializer(PopulationInitializer):
    def __init__(self, population_size, population_bits):
        super().__init__(population_size)
        self.population_bits = population_bits

    #Generates a single binary individual
    def generate_individual(self):
        bit_string = list()
        for i in range(self.population_bits):
            bit_string.append(1 if random.random() > 0.5 else 0)
        return bit_string
        
    #Generates a population of random binary individuals
    def generate_population(self):
        population = list()
        for i in range(self.population_size):
            population.append(self.generate_individual())
        return population 
        


