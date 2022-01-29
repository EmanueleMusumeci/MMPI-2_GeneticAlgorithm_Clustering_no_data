#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) 
 
 Early stopping abstract class and simple early stopping method that
 terminates training after a specific number of iterations with no
 improvement

"""
#######################################################################
import abc

import numpy as np
from numpy import random

#Base abstract class for early stopping methods
class EarlyStoppingCriterion(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, individual):
        pass

#This early stopping method determines if there wasn't any improvement for a specific
#number of turns, in which case it terminates training
class ImprovementHistoryWithPatience(EarlyStoppingCriterion):
    def __init__(self, patience):
        self.patience = patience
        self.previous_best_value = 0
        self.iterations_since_previous_best_value = 0

    def __call__(self, current_fitness_value):
        if current_fitness_value > self.previous_best_value:
            self.previous_best_value = current_fitness_value
            self.iterations_since_previous_best_value = 0
        else:
            self.iterations_since_previous_best_value += 1

        if self.iterations_since_previous_best_value > self.patience:
            print("Early stopped after "+str(self.iterations_since_previous_best_value)+" iterations without improvement (patience: "+str(self.patience)+")")
            return True
        else:
            return False        
