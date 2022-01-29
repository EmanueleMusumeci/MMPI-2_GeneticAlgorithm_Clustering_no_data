##########################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) 
 
 This file contains a simple implementation of an optimizer (a simple one)
 that perform all basic steps of a vanilla genetic algorithm.

"""
##########################################################################

import math

import dill

from modules.mutation import MutationMethod
from modules.population_initialization import *
from modules.fitness_evaluation import *
from modules.crossover import *
from modules.clustering import *
from modules.visualization import *
from modules.early_stopping import *

import random

"""
GeneticBinaryOptimizer class
"""
class GeneticBinaryOptimizer:

    """
    Parameters:
    - initial_generation_method: PopulationInitializer instance used to generate the initial population
    - fitness_evaluation_method: FitnessEvaluationMethod instance wrapping a fitness function for the current optimization problem
    - crossover_method: CrossoverMethod instance used for parents crossover to generate the next generation
    - mutation_method: MutationMethod instance used to mutate each new generation individual
    - early_stopping_criterion: EarlyStoppingCriterion instance that stops learning when no improvements are seen after a certain amount of turns
    - result_visualization_method: VisualizationMethod instance that shows the current improvements (if there are any) with custom metrics
    - n_parent_candidates: pool size of candidates during the tournament selection step

    - NOTICE: the population is initialized at construction time
    """
    def __init__(self, 
                initial_generation_method,
                fitness_evaluation_method, 
                crossover_method, 
                mutation_method, 
                early_stopping_criterion = None, 
                result_visualization_method = None,
                n_parent_candidates = 3,
                checkpoints_dir = None
                ):

        self.initial_generation = initial_generation_method
        assert isinstance(initial_generation_method, PopulationInitializer)

        self.fitness_evaluation = fitness_evaluation_method
        assert isinstance(fitness_evaluation_method, FitnessEvaluationMethod)

        self.crossover = crossover_method
        assert isinstance(crossover_method, CrossoverMethod)

        self.mutation = mutation_method
        assert isinstance(mutation_method, MutationMethod)

        self.early_stopping_criterion = early_stopping_criterion
        if self.early_stopping_criterion is not None:
            assert isinstance(early_stopping_criterion, EarlyStoppingCriterion)

        self.result_visualization = result_visualization_method
        assert isinstance(result_visualization_method, VisualizationMethod)
        
        self.population = [] 
        initial_population = self.initial_generation.generate_population()
        initial_population_fitness = [self.fitness_evaluation(element) for element in initial_population]
        self.population.append(
            {
                "elements" : initial_population,
                "fitness" : initial_population_fitness,
                "best_element" : self.select_best(initial_population, initial_population_fitness),
                "best_fitness" : max(initial_population_fitness)
            }
        )

        self.current_iteration = 0

        self.n_parent_candidates = n_parent_candidates
        #self.save_checkpoints_every = save_checkpoints_every
        self.checkpoints_dir = checkpoints_dir

    # Returns the current population size   
    def __len__(self):
        return len(self.population[self.current_iteration]["elements"])    

    # Select the best element in the batch based on th fitness values
    def select_best(self, batch, fitness_values):
        return [i for _,i in reversed(sorted(zip(fitness_values, batch)))][0]

    # Selects randomly k elements and then return the one with the best fitness value
    def tournament_selection(self ,batch, fitness_values, select_k, select_best_k = False): 
        assert select_k < len(batch), "Wrong subset size for tournament selection: batch size is "+str(len(batch))+" while selection size is "+str(select_k)
        
        if select_best_k:
            selected_batch = batch
            selected_batch_fitness_values = fitness_values
        else:
            selected_batch_indices = random.sample(range(len(batch)), select_k)
            selected_batch = [batch[i] for i in selected_batch_indices]
            selected_batch_fitness_values = [fitness_values[i] for i in selected_batch_indices]

        #Sort the batch in descending order based on its fitness values
        selected_batch_sorted_on_fitness_values = [i for _,i in reversed(sorted(zip(selected_batch_fitness_values, selected_batch)))]
        
        #print(sorted(zip(fitness_values, selected_batch)))
        return selected_batch_sorted_on_fitness_values[0]

    # Prints the current population (TESTING ONLY)
    def print_population(self, pop, values):
        for i, (el, fit) in enumerate(zip(pop, values)):
            print("\tElement #"+str(i)+": "+str(el))
            print("\t\tFitness: "+str(fit)+"\n")

    # Saves a checkpoint for the current state of the optimizer
    def save_snapshot(self):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        
        complete_file_path = os.path.join(self.checkpoints_dir,"model")
        
        #Save snapshot
        with open(complete_file_path+".pt", mode="wb") as f:
            dill.dump(self,f)
        
        #write a small log file for best element scores
        with open(complete_file_path+".scores", mode="w") as f:
            for pop_iteration in self.population:
                best_element_str = ""
                for item in pop_iteration["best_element"]:
                    best_element_str+=str(item)
                f.write("Best element:\n"+best_element_str+"\nFitness:\n"+str(pop_iteration["best_fitness"])+"\n")

    @classmethod
    # Loads a checkpoint and restores it as the current state of the optimizer
    def load(cls, model_dir):
        complete_file_path = os.path.join(model_dir,"model.pt")
        with open(complete_file_path, mode = "rb") as f:
            instance = dill.load(f)
        return instance

    """
    The genetic optimization process performs the following steps: 
       0) Compute fitness for all individuals of initial population using the specified fitness evaluation function
       LOOP for n_iterations
            1) Select population_size parents
            2) Draw population_size/2 parents pairs
            3) Generate two children from each pair using the specified crossover method
            4) Perform random mutations on the population using the chosen method
            5) Compute fitness for each element
            6) If there was an improvement visualize it using the chosen visualization methods
            7) If required, save a snapshot
            8) If there wasn't any improvement for a certain amount of iterations, break the training loop
        END LOOP
    """
    def learn(self, n_iterations):

        #Evolutionary loop
        #0) Compute fitness for each element
        previous_best_value = max(self.population[self.current_iteration]["fitness"])
        
        if self.result_visualization is not None:
            #Visualize the clustering on the dataset with the original number of dimensions
            self.result_visualization(self.population[self.current_iteration]["best_element"], self.current_iteration, 
                                        file_name_prefix = "transformed_",
                                        figure_title = type(self.fitness_evaluation).__name__ + " optimization, " + \
                                            type(self.fitness_evaluation.clusterer).__name__ + " clustering, " + \
                                            "\nIteration "+str(self.current_iteration),
                                            figure_caption = "Fitness: " + str(previous_best_value),
                                            show_original_points = False,
                                            view_clusters = False)
            #Visualize the clustering on the dataset with the selected number of dimensions
            self.result_visualization(self.population[self.current_iteration]["best_element"], self.current_iteration, 
                                        file_name_prefix = "original_",
                                        figure_title = type(self.fitness_evaluation).__name__ + " optimization, " + \
                                            type(self.fitness_evaluation.clusterer).__name__ + " clustering, " + \
                                            "\nIteration "+str(self.current_iteration),
                                            figure_caption = "Fitness: " + str(previous_best_value),
                                            show_original_points = True)

        for iteration in range(n_iterations):
            print("Iteration "+str(self.current_iteration))
            #print_population(population, self.population[self.current_iteration]["fitness"])
            
            parents = list()
            #1) Select population_size parents ("tournament selection")
            for _ in range(len(self.population[self.current_iteration]["elements"])):
                #1.1) Draw randomly n_draw parent candidates from the population
                selected_parent = self.tournament_selection(self.population[self.current_iteration]["elements"], self.population[self.current_iteration]["fitness"], self.n_parent_candidates)
                #1.2) Select the element with best fitness as a parent (first element of the selected batch which is ordered in descending order)
                parents.append(selected_parent)            

            #2) Draw randomly population_size/2 pairs
            # If the population size is odd, extract population_size-1 samples 
            # (this can happen only in the first evolutionary iteration)
            odd_population = len(self.population[self.current_iteration]["elements"])%2 == 1

            #Select parents pairs
            parents_pairs = []
            for i in range(0, len(self.population[self.current_iteration]["elements"])-1 if odd_population else len(self.population[self.current_iteration]["elements"]), 2):
                parents_pairs.append((parents[i], parents[i+1]))

            #3) Generate two children from each pair using bit crossover (or other operations)
            children = list()
            for pair in parents_pairs:
                crossover_tuple = self.crossover(pair[0], pair[1])
                children.append(crossover_tuple[0])
                children.append(crossover_tuple[1])
            
            #3.1) If the population size is an odd number, perform one more crossover among individuals of a parents couple, 
            #and choose randomly which child to add 
            if odd_population:
                pair = random.sample(self.population[self.current_iteration]["elements"], 2)
                crossover_tuple = self.crossover(pair[0], pair[1])
                if random.random() > 0.5:
                    children.append(crossover_tuple[0])
                else:
                    children.append(crossover_tuple[1])

            #4) Apply a random mutation
            for i in range(len(self.population[self.current_iteration]["elements"])):
                children[i] = self.mutation(children[i])

            #5) Compute fitness for each child element
            children_fitness = [self.fitness_evaluation(element) for element in children]

            self.population.append(
                {
                    "elements" : children,
                    "fitness" : children_fitness,
                    "best_element" : self.select_best(children, children_fitness),
                    "best_fitness" : max(children_fitness),
                }
            )
            self.current_iteration = self.current_iteration + 1

            #6) If there was an improvement visualize it using the chosen visualization methods
            current_best_value = max(children_fitness)
            if current_best_value > previous_best_value:
                best_element = self.select_best(children, children_fitness)
                print("New best: "+str(best_element)+", Fitness: "+str(current_best_value))
                previous_best_value = current_best_value

                if self.result_visualization is not None:
                    self.result_visualization(best_element, self.current_iteration, 
                                              figure_title = type(self.fitness_evaluation).__name__ + " optimization, " + \
                                                  type(self.fitness_evaluation.clusterer).__name__ + " clustering, " + \
                                                  "\nIteration "+str(self.current_iteration),
                                                figure_caption = "Fitness: " + str(current_best_value),
                                            show_original_points = True)
                    self.result_visualization(best_element, self.current_iteration, 
                                              file_name_prefix = "transformed_",
                                              figure_title = type(self.fitness_evaluation).__name__ + " optimization, " + \
                                                  type(self.fitness_evaluation.clusterer).__name__ + " clustering, " + \
                                                  "\nIteration "+str(self.current_iteration),
                                                figure_caption = "Fitness: " + str(current_best_value),
                                            show_original_points = False,
                                            view_clusters = False)
                    
            elif self.current_iteration == 0 and self.result_visualization is not None:
                self.result_visualization(best_element, self.current_iteration, 
                                            figure_title = type(self.fitness_evaluation).__name__ + " optimization, " + \
                                                type(self.fitness_evaluation.clusterer).__name__ + " clustering, " + \
                                                "\nIteration "+str(self.current_iteration),
                                                figure_caption = "Fitness: " + str(current_best_value),
                                            show_original_points = True)
                self.result_visualization(best_element, self.current_iteration, 
                                            file_name_prefix = "transformed_", 
                                            figure_title = type(self.fitness_evaluation).__name__ + " optimization, " + \
                                                type(self.fitness_evaluation.clusterer).__name__ + " clustering, " + \
                                                "\nIteration "+str(self.current_iteration),
                                                figure_caption = "Fitness: " + str(current_best_value),
                                            show_original_points = False,
                                            view_clusters = False)

            #7) If required, save a snapshot
            if self.checkpoints_dir is not None:
                self.save_snapshot()

            #8) If there wasn't any improvement for a certain amount of iterations, break the training loop
            if self.early_stopping_criterion is not None and self.early_stopping_criterion(current_best_value):
                break
            
        #print(self.population[self.current_iteration]["elements"])

        #Selects and returns the best model
        return self.population[self.current_iteration]["best_element"]
