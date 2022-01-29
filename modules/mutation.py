#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) #
 
 Abstract class for a generic mutation method and vanilla random bit 
 flip mutation

"""
#######################################################################
import abc

import numpy as np
from numpy import random

#Base abstract class for mutation methods, that mutate genetic characteristics of a single individual
class MutationMethod(metaclass=abc.ABCMeta):
    def __init__(self, mutation_probability):
        self.mutation_probability = mutation_probability

    @abc.abstractmethod
    def __call__(self, element):
        pass

#Mutate element by flipping bits with a probability of bit_flip_probability
class RandomBitFlip(MutationMethod):
    def __init__(self, mutation_probability):
        super().__init__(mutation_probability)
        self.bit_flip_probability = self.mutation_probability

    def __call__(self, element):

        for i,bit in enumerate(element):
            if random.random() <= self.bit_flip_probability:
                element[i] = 1 if bit==1 else 0

        return element
        
