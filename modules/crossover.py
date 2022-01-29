#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) #
 
 Abstract class for crossover methods and vanilla random split, splitting both parents
 with a probability crossover_probability, at a random point in the string

"""
#######################################################################
import numpy as np

import random

import abc

#Base abstract class for crossover methods, that combine two parents into two children
class CrossoverMethod(metaclass=abc.ABCMeta):
    def __init__(self, crossover_probability):
        self.crossover_probability = crossover_probability

    @abc.abstractmethod
    def __call__(self, parent1, parent2):
        pass

#The child will have as a first slice the bits of the first parent and as a second slice, bits from the second parent
#The split point is chosen randomly (but the beginning/end of the string are excluded)
class RandomSplit(CrossoverMethod):
    def __init__(self, crossover_probability):
        super().__init__(crossover_probability)

    def __call__(self, parent1, parent2):
        assert len(parent1) == len(parent2), "Parents have different length"
        
        if random.random() <= self.crossover_probability:
            #crossover_point = math.ceil(len(parent1)/2)
            crossover_point = np.random.randint(1, len(parent1)-2)
            return (parent1[:crossover_point] + parent2[crossover_point:], 
                    parent2[:crossover_point] + parent1[crossover_point:])
        else:
            return parent1.copy(), parent2.copy()
