#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) 
 
 Implementations for various clustering methods, an abstract class for a Clusterer
 and some wrappers for these methods

"""
#######################################################################

import numpy as np
from scipy.sparse.construct import random
from sklearn import cluster

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import sklearn.preprocessing


import scipy

import abc


#Normalizes data
def normalize(data, mean, stddev):
    return sklearn.preprocessing.normalize(data)

#PCA used as a dimensionality reduction (for visual representations)
def get_PCA_fit_model(points, components = 3, random_state = 0):
    pca = PCA(n_components=3, random_state = random_state)
    pca.fit(points)
    return pca

#PCA used as a dimensionality reduction (for visual representations)
def transform_PCA(points, components = 3):
    pca = get_PCA_fit_model(points, components=components)
    points = pca.transform(points)
    return points, pca

'''
def nearest_neighbors(points):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    print(distances)
    print(indices)

def nearest_neighbors_clustering(points, neighbors = 1):
    model = KNeighborsClassifier(n_neighbors = neighbors)
    model.fit(x_training_data, y_training_data)
'''

#Apply K-Means clustering and extract centroids
def kmeans(data, clusters, random_state = None):
    if random_state is None:
        model = KMeans(n_clusters = clusters)
    else:
        #print("RANDOM STATE "+ str(random_state))
        model = KMeans(n_clusters = clusters, random_state=random_state)
    model.fit(data)
    return model.labels_, model.cluster_centers_

#Apply Gaussian Mixture Model (GMM) clustering and extract centroids
def gmm(data, clusters, random_state = None):

    if random_state is None:
        #gmm = GaussianMixture(n_components=np.shape(points)[1]).fit(data)
        gmm = GaussianMixture(n_components=clusters).fit(data)
    else:
        #gmm = GaussianMixture(n_components=np.shape(points)[1]).fit(data)
        gmm = GaussianMixture(n_components=clusters, random_state=random_state).fit(data)

    # assign a cluster to each example
    gmm_predictions = gmm.predict(data)
    
    #Taken from https://stackoverflow.com/questions/47412749/how-can-i-get-a-representative-point-of-a-gmm-cluster
    gmm_centroids = np.empty(shape=(gmm.n_components, data.shape[1]))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
        gmm_centroids[i, :] = data[np.argmax(density)]
    
    return gmm_predictions, gmm_centroids

#Apply spectral clustering, a non-linear clustering method, with Radial Basis Functions
#Notice: doesn't make sense to return centroids as clustering might be non-linear
def spectral_clustering_rbf(data, clusters = 2, random_state = None):
    if random_state is None:
        spec = SpectralClustering(assign_labels='discretize', n_clusters=clusters, affinity="rbf")
    else:
        spec = SpectralClustering(assign_labels='discretize', n_clusters=clusters, affinity="rbf", random_state=random_state)
        
    spec_predictions = spec.fit_predict(data)

    return spec_predictions, None

#Apply spectral clustering, a non-linear clustering method, using Nearest Neighbors
#Notice: doesn't make sense to return centroids as clustering might be non-linear
def spectral_clustering_nn(data, clusters = 2, random_state = None):
    if random_state is None:
        spec = SpectralClustering(assign_labels='discretize', n_clusters=clusters, affinity="nearest_neighbors")
    else:
        spec = SpectralClustering(assign_labels='discretize', n_clusters=clusters, affinity="nearest_neighbors", random_state=random_state)
        
    spec_predictions = spec.fit_predict(data)

    return spec_predictions, None
#-----------------

#Clusterer abstract class
class Clusterer:
    def __init__(self, clusters):
        self.clusters = clusters
    
    @abc.abstractmethod
    def __call__(self):
        pass

class MMPI2Clusterer(Clusterer):
    def __init__(self, dataset, clusters, use_couples = False, use_groups = False, normalize_data = False):
        super().__init__(clusters)
        self.dataset = dataset
        self.use_couples = use_couples
        self.use_groups = use_groups
        self.normalize_data = normalize_data
    
    @abc.abstractmethod
    def __call__(self):
        pass
        
    def get_selected_fields(self, selected_data_fields = None, return_transformed_points=False):
        if selected_data_fields is None:
            if self.use_groups:
                selected_data_fields = [1] * len(self.dataset.get_psychometric_scales_groups_indices())
            else:
                selected_data_fields = [1] * len(self.dataset.get_psychometric_scales_indices())

        #Check if string contains only '0'
        if not np.any(selected_data_fields):
            raise ValueError

        if return_transformed_points:
            if self.use_couples:
                #Compute differences of values from specific scales and convert them to a numpy array
                if self.use_groups:
                    points = self.dataset.binary_array_to_psy_scale_groups_couple_difference_to_numpy_array(selected_data_fields)
                else:
                    points = self.dataset.binary_array_to_psy_scales_couple_difference_to_numpy_array(selected_data_fields)
            else:
                #Return values from specific scales and convert them to a numpy array
                if self.use_groups:
                    points = self.dataset.binary_array_to_psy_scale_groups_to_numpy_array(selected_data_fields)
                else:
                    points = self.dataset.binary_array_to_psy_scales_to_numpy_array(selected_data_fields)
        else:
            if self.use_groups:
                points = self.dataset.binary_array_to_psy_scale_groups_to_numpy_array(selected_data_fields)
            else:    
                points = self.dataset.binary_array_to_psy_scales_to_numpy_array(selected_data_fields)

        if self.normalize_data:
            points = normalize(points, mean=50, stddev=10)

        return points    

#1) If use_couples is False: 
#       uses binary_string to select specific data fields
#   else:
#        Computes differences in psychometric values between components of a same couple, using binary_string to select specific data fields
#2) Clusters data points using KMeans algorithm
class MMPI2_KMeans_clusterer(MMPI2Clusterer):
    def __init__(self, dataset, clusters, use_couples = False, use_groups=False, normalize_data = True, use_random_seed = False):
        super().__init__(dataset, clusters, use_couples = use_couples, use_groups = use_groups, normalize_data = normalize_data)
        if use_random_seed:
            self.random_seed = np.random.randint(0,9999)
        else:
            self.random_seed = None
    
    def __call__(self, selected_data_fields = None, return_transformed_points = False, ground_truth = None):
        
        original_points = self.get_selected_fields(return_transformed_points=return_transformed_points)
        transformed_points = self.get_selected_fields(selected_data_fields=selected_data_fields, return_transformed_points=return_transformed_points)

        #Apply K-Means clustering with 2 clusters (supposedly honest/lying) and extract centroids
        k_means_transformed_predictions, k_means_transformed_centroids = kmeans(transformed_points, clusters = self.clusters, random_state = self.random_seed)

        if return_transformed_points:
            points = transformed_points
            centroids = k_means_transformed_centroids
        else:
            #Also apply k_means to the original distribution to extract the centroids
            #   NOTICE: using the same random state as before ensures that the cluster prediction labels are the same
            #   for predictions and centroids ('0' -> first cluster -> first centroid etc.) 
            #   so that centroid colors are the same as data point colors in the visual representation
            _, centroids = kmeans(original_points, clusters = self.clusters, random_state = self.random_seed)
            points = original_points

        if ground_truth is not None:
            cluster_predictions = ground_truth
        else:
            cluster_predictions = k_means_transformed_predictions

        return points, cluster_predictions, centroids

#1) If use_couples is False: 
#       uses binary_string to select specific data fields
#   else:
#        Computes differences in psychometric values between components 
#2) Clusters data points using GMM algorithm
class MMPI2_GMM_clusterer(MMPI2Clusterer):
    def __init__(self, dataset, clusters, use_couples = False, use_groups = False, normalize_data = True, use_random_seed = False):
        super().__init__(dataset, clusters, use_couples = use_couples, use_groups = use_groups, normalize_data = normalize_data)
        if use_random_seed:
            self.random_seed = np.random.randint(0,9999)
        else:
            self.random_seed = None

    def __call__(self, selected_data_fields = None, return_transformed_points = False, ground_truth = None):

        original_points = self.get_selected_fields()
        transformed_points = self.get_selected_fields(selected_data_fields=selected_data_fields)

        #Apply GMM clustering with 2 clusters and extract centroids from both original distribution
        #and projected distribution (this only to extract projected centroids)
        gmm_transformed_predictions, gmm_transformed_centroids = gmm(transformed_points, clusters = self.clusters, random_state = self.random_seed)

        if return_transformed_points:
            points = transformed_points
            centroids = gmm_transformed_centroids
        else:
            #Also apply GMM to the original distribution to extract the centroids
            #   NOTICE: using the same random state as before ensures that the cluster prediction labels are the same
            #   for predictions and centroids ('0' -> first cluster -> first centroid etc.) 
            #   so that centroid colors are the same as data point colors in the visual representation
            _, centroids = gmm(original_points, self.clusters, random_state = self.random_seed)
            points = original_points

        if ground_truth is not None:
            cluster_predictions = ground_truth
        else:
            cluster_predictions = gmm_transformed_predictions

        return points, cluster_predictions, centroids

#1) If use_couples is False: 
#       uses binary_string to select specific data fields
#   else:
#        Computes differences in psychometric values between components 
#2) Clusters data points using the Spectral clustering algorithm
class MMPI2_Spectral_clusterer(MMPI2Clusterer):
    def __init__(self, dataset, clusters, use_couples = False, use_groups = False, normalize_data = True, use_rbf = False):
        super().__init__(dataset, clusters, use_couples = use_couples, use_groups = use_groups, normalize_data = normalize_data)
        self.use_rbf = use_rbf

    def __call__(self, selected_data_fields = None, return_transformed_points = False, ground_truth = None):

        original_points = self.get_selected_fields()
        transformed_points = self.get_selected_fields(selected_data_fields=selected_data_fields)

        if self.use_rbf:
            spectral_predictions, _ = spectral_clustering_rbf(transformed_points, self.clusters)
        else:
            spectral_predictions, _ = spectral_clustering_nn(transformed_points, self.clusters)

        if return_transformed_points:
            points = transformed_points
        else:
            points = original_points

        if ground_truth is not None:
            cluster_predictions = ground_truth
        else:
            cluster_predictions = spectral_predictions

        return points, cluster_predictions, _
