#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) 
 
 Implementations for various fitness evaluation functions wrappers

"""
#######################################################################
import abc

import numpy as np
import scipy
import sklearn

from itertools import product

#Base abstract class for crossover methods, that combine two parents into two children
class FitnessEvaluationMethod(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self.name = name
    
    @abc.abstractmethod
    def __call__(self, individual):
        pass

#This fitness evaluation method does the following:
#1) Clusters data points using the given clusterer
#2) Computes accuracy of the classification given the ground truth cluster labels
class ClusterAccuracy(FitnessEvaluationMethod):
    def __init__(self, clusterer, ground_truth_cluster_labels):
        super().__init__("ClusterAccuracy")
        self.clusterer = clusterer
        self.ground_truth_cluster_labels = ground_truth_cluster_labels

    def __call__(self, individual):
        
        try:
            #Perform clustering using the given clusterer
            _, predictions, _ = self.clusterer(individual)
        except ValueError:
            return 0
            
        return sklearn.metrics.accuracy_score(self.ground_truth_cluster_labels, predictions)

#This fitness evaluation method does the following:
#1) Clusters data points using the given clusterer
#2) Computes the distance between centroids of these clusters
class ClusterCentroidDistance(FitnessEvaluationMethod):
    def __init__(self, clusterer):
        super().__init__("ClusterCentroidDistance")
        self.clusterer = clusterer

    def __call__(self, individual):
        
        try:
            #Perform clustering using the given clusterer
            _, _, cluster_centroids = self.clusterer(individual, return_transformed_points = True)
        except ValueError:
            return 0
        
        centroid_distances = []
        for i in range(len(cluster_centroids)):
            for j in range(i+1, len(cluster_centroids)):
                centroid_distances.append(np.linalg.norm(cluster_centroids[i]-cluster_centroids[j]))

        return min(centroid_distances)

#This fitness evaluation method does the following:
#1) Clusters data points using the given clusterer
#2) Computes the minimum distance between data points (the higher the better)
class ClusterMinimumPointDistance(FitnessEvaluationMethod):
    def __init__(self, clusterer):
        super().__init__("ClusterMinimumPointDistance")
        self.clusterer = clusterer

    def __call__(self, individual):
        
        try:
            #Perform clustering using the given clusterer
            points, cluster_predictions, _ = self.clusterer(individual, return_transformed_points = True)
        except ValueError:
            return 0
        
        min_overall_distance = None
        labels = np.unique(cluster_predictions)
        #For each pair of clusters
        for (label1, label2) in product(labels, repeat=2):
            if label1 == label2:
                continue
            
            #Extract points belonging to cluster 1
            points_with_label1 = points[np.where(cluster_predictions == label1)]
            #print(points_with_label1)

            #Extract points belonging to cluster 2
            points_with_label2 = points[np.where(cluster_predictions == label2)]
            #print(points_with_label2)

            #Compute minimum of distances between the two sets of points
            min_distance = np.min(scipy.spatial.distance.cdist(points_with_label1, points_with_label2))
            
            #Compute minimum overall distance
            if min_overall_distance is None or min_overall_distance > min_distance:
                min_overall_distance = min_distance

        return min_overall_distance

#This fitness evaluation method does the following:
#1) Clusters data points using the given clusterer
#2) Computes the average distance between data points and the centroid of their
#   assigned cluster (the higher the better)
class ClusterAveragePointDistanceFromCentroid(FitnessEvaluationMethod):
    def __init__(self, clusterer):
        super().__init__("ClusterAveragePointDistanceFromCentroid")
        self.clusterer = clusterer

    def __call__(self, individual):
        
        try:
            #Perform clustering using the given clusterer
            points, cluster_predictions, _ = self.clusterer(individual, return_transformed_points = True)
        except ValueError:
            return 0
        
        min_overall_distance = None
        labels = np.unique(cluster_predictions)
        #For each pair of clusters
        for (label1, label2) in product(labels, repeat=2):
            if label1 == label2:
                continue
            
            #Extract points belonging to cluster 1
            points_with_label1 = points[np.where(cluster_predictions == label1)]
            #print(points_with_label1)

            #Extract points belonging to cluster 2
            points_with_label2 = points[np.where(cluster_predictions == label2)]
            #print(points_with_label2)

            #Compute minimum of distances between the two sets of points
            min_distance = np.min(scipy.spatial.distance.cdist(points_with_label1, points_with_label2))
            
            #Compute minimum overall distance
            if min_overall_distance is None or min_overall_distance > min_distance:
                min_overall_distance = min_distance

        return min_overall_distance

#Fitness function for the OneMax problem (obtaining a binary string of only 1s) which counts the number of 1s
class OneMax(FitnessEvaluationMethod):
    def __init__(self, name):
        super().__init__("OneMax")

    def __call__(self, individual):
        return sum(individual)
