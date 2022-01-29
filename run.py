'''
TODO

DONE 1) Dataset cleanup
    1b) Might find a way to determine if the dataset contains a gaussian noise
DOING 2) Clustering
CHECK IF NEEDED 3) Cluster classification (?)
CHECK IF NEEDED 4) Cluster profiling to correlate cluster with boolean questionnaire (might use a neural network here or a different model)
CHECK IF NEEDED 5) Extract features to determine the cluster given the questionnaire
6) Try including the questionnaires as well or only questionnaires
7) Add iteration number to viz
'''

'''
TODO Optimization
DONE 1) Ability to restore population and parameters of genetic algorithm
DONE 2) Save each best element of population and then build a gif of all 3D/2D point clouds
3) Perfect the fitness function
DONE 4) Save best element history
5) Fix couples (always same order in M or F, also handling homosexual couples)
'''

import os

import numpy as np
from modules.clustering import MMPI2_GMM_clusterer, MMPI2_KMeans_clusterer, MMPI2_Spectral_clusterer
from modules.crossover import RandomSplit
from modules.early_stopping import ImprovementHistoryWithPatience
from modules.fitness_evaluation import ClusterCentroidDistance, ClusterMinimumPointDistance
from modules.mutation import RandomBitFlip
from modules.population_initialization import BinaryPopulationInitializer
from modules.visualization import Cluster3DVisualizer, Cluster2DVisualizer, Cluster2D3DVisualizer
from modules.dataset import MMPI2Dataset, MMPI2InterviewSample
from modules.optimization import GeneticBinaryOptimizer

if __name__=="__main__":
    BASE_DIR = os.path.dirname(__file__)
    #print("Base directory: "+BASE_DIR)

    #DATASET_DIR = os.path.join(BASE_DIR,"data")
    SNAPSHOT_DIR = "snapshots"
    
    DATASET_DIR = "data"
    print("Dataset directory: "+DATASET_DIR)

    dataset = MMPI2Dataset.load_dataset(DATASET_DIR, "MMPI2AdoptionData.csv")
    N_CLUSTERS = 4

    POPULATION_SIZE = 300
    #STRING_SIZE = len(["validity", "clinical", "content", "supplemental", "psy5", "other"])
    N_ITERATIONS = 500
    CROSSOVER_PROBABILITY = 0.9
    TOURNAMENT_SELECTION_CANDIDATES = 20

    EARLY_STOPPING_PATIENCE = 10
    
#####################
###   Clustering  ###
#####################
    CLUSTERING_METHOD = "KMEANS"
    #CLUSTERING_METHOD = "GMM"
    #CLUSTERING_METHOD = "SPECTRAL"
    USE_COUPLES = False
    USE_PSY_SCALES_GROUPS = False
    INCLUDE_ITEMS = False
    NORMALIZE = False
    
    #VISUALIZE = "2D"
    #VISUALIZE = "3D"
    VISUALIZE = "2D3D"

    FITNESS_EVALUATION_METHOD = "CentroidDistance"
    #FITNESS_EVALUATION_METHOD = "MinimumPointDistance"

    MODEL_NAME = CLUSTERING_METHOD + "_" + str(N_CLUSTERS) + "_clusters" + ("_Couples" if USE_COUPLES else "_Single_individual") + ("_Scale_groups" if USE_PSY_SCALES_GROUPS else "_Single_scales") + ("_"+FITNESS_EVALUATION_METHOD) + ("_normalized" if NORMALIZE else "") + ("_With_Items" if INCLUDE_ITEMS else "")
    print(MODEL_NAME)

    SAVE_IMAGES_TO_DIR = os.path.join(SNAPSHOT_DIR, MODEL_NAME, "images")
    #SAVE_CHECKPOINTS_TO_DIR = os.path.join(SNAPSHOT_DIR, MODEL_NAME, "checkpoints")
    SAVE_CHECKPOINTS_TO_DIR = os.path.join(SNAPSHOT_DIR, MODEL_NAME)

### CLUSTERING ###
    if CLUSTERING_METHOD == "KMEANS":
#NOTICE: gives the "numpy.linalg.LinAlgError: singular matrix" error
        clusterer = MMPI2_KMeans_clusterer(
            dataset, 
            N_CLUSTERS,
            use_couples = USE_COUPLES,
            use_groups=USE_PSY_SCALES_GROUPS,
            normalize_data=NORMALIZE,
            use_random_seed=True
        )
    elif CLUSTERING_METHOD == "GMM":
        if USE_COUPLES:
            print("GMM requires USE_COUPLES to be False")
            exit()
        else:
            clusterer = MMPI2_GMM_clusterer(
                    dataset, 
                    N_CLUSTERS,
                    use_couples = USE_COUPLES,
                    use_groups=USE_PSY_SCALES_GROUPS,
                    normalize_data=NORMALIZE
                )
    elif CLUSTERING_METHOD == "SPECTRAL":
        clusterer = MMPI2_Spectral_clusterer(
                dataset, 
                N_CLUSTERS,
                use_couples = USE_COUPLES,
                use_groups=USE_PSY_SCALES_GROUPS,
                normalize_data=NORMALIZE
            )

    if USE_PSY_SCALES_GROUPS:
        STRING_SIZE = len(["validity", "clinical", "content", "supplemental", "psy5", "other"])
    else:
        STRING_SIZE = len(dataset.get_psychometric_scales_indices())

    #MUTATION_PROBABILITY = 1.0/STRING_SIZE
    MUTATION_PROBABILITY = 0.8


    if NORMALIZE:
        SMALLER_SCALE_RANGE = (1, 1, 1)
        LARGER_SCALE_RANGE = (3, 3, 3)
    else:
        SMALLER_SCALE_RANGE = (150, 150, 150)
        LARGER_SCALE_RANGE = (150, 150, 150)

    if VISUALIZE == "2D":
        visualizer_method = Cluster2DVisualizer(
            clusterer, 
            show=False, 
            save_snapshot_images_to=SAVE_IMAGES_TO_DIR,
            smaller_scale_range=SMALLER_SCALE_RANGE,
            larger_scale_range=LARGER_SCALE_RANGE
            )
    elif VISUALIZE == "3D":
        visualizer_method = Cluster3DVisualizer(
            clusterer, 
            show=False, 
            save_snapshot_images_to=SAVE_IMAGES_TO_DIR,
            smaller_scale_range=SMALLER_SCALE_RANGE,
            larger_scale_range=LARGER_SCALE_RANGE
            )
    elif VISUALIZE == "2D3D":
        visualizer_method = Cluster2D3DVisualizer(
            clusterer, 
            show=False, 
            save_snapshot_images_to=SAVE_IMAGES_TO_DIR,
            smaller_scale_range=SMALLER_SCALE_RANGE,
            larger_scale_range=LARGER_SCALE_RANGE
            )

    if FITNESS_EVALUATION_METHOD == "CentroidDistance":
        fitness_function = ClusterCentroidDistance(
            clusterer
        )
    elif FITNESS_EVALUATION_METHOD == "MinimumPointDistance":
        fitness_function = ClusterMinimumPointDistance(
            clusterer
        )

#####################
###    Learner    ###
#####################

    visualizer_method([1]*(len(dataset.get_psychometric_scales_groups_indices()) if USE_PSY_SCALES_GROUPS else len(dataset.get_psychometric_scales_indices())),
                            "original", 
                    figure_title = type(fitness_function).__name__ + " optimization, " + \
                        "Initial clusters", show_original_points=True)

    genetic_learner = GeneticBinaryOptimizer(
        BinaryPopulationInitializer(POPULATION_SIZE, STRING_SIZE),
        fitness_function,
        RandomSplit(CROSSOVER_PROBABILITY),
        RandomBitFlip(MUTATION_PROBABILITY),
        early_stopping_criterion=ImprovementHistoryWithPatience(EARLY_STOPPING_PATIENCE),
        #early_stopping_criterion=None,
        result_visualization_method=visualizer_method,
        n_parent_candidates=min(TOURNAMENT_SELECTION_CANDIDATES, POPULATION_SIZE-1),
        checkpoints_dir=SAVE_CHECKPOINTS_TO_DIR
    )

    genetic_learner.learn(N_ITERATIONS)
    
    genetic_learner.result_visualization.generate_gif()
    
    genetic_learner.result_visualization.generate_final_comparison()
