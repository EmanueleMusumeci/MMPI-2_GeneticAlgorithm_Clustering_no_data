#######################################################################
"""
 @author: Emanuele Musumeci (https://github.com/EmanueleMusumeci) 
 
 Implementations of visualization functions for OneMax (simply prints the string),
 2D and 3D data points distributions (or both 2D and 3D) with their respective 
 centroid highlighted (not a data point), that also save the rendered visualizations to file 
 and render a GIF animation from all these visualizations

"""
#######################################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from numpy.core import overrides
from sklearn import cluster

from modules.clustering import transform_PCA

import numpy as np
import imageio

import natsort

import abc
import os
import shutil

def generate_cluster_color_palette(n_clusters, cluster_points_alpha = 0.8, cluster_centroids_alpha = 1):
    return plt.cm.brg(np.linspace(0, 1, n_clusters),alpha=cluster_points_alpha), \
        plt.cm.brg(np.linspace(0, 1, n_clusters),alpha=cluster_centroids_alpha) #Equally spaced colors
        
def visualize_2D(points, show = False, save_to_dir = None, 
                cluster_predictions = None, cluster_centroids = None, 
                file_name = "figure", figure_title = None, figure_caption = None,
                scale_range = None, cluster_color_palette = None, centroid_color_palette = None):
    
    if scale_range is not None:
        assert len(scale_range) >= 2, "Wrong scale_range length: "+str(len(scale_range)) + " (should be 2 or more)"

    #2D visualization
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    plt.cla()

    #Set figure font
    plt.rcParams.update({'font.size': 7})
    #Set figure caption
    if figure_caption is not None:
        fig.text(.5, .05, figure_caption, ha='center')

    #Set figure font
    plt.rcParams.update({'font.size': 5})
    #Set title
    if figure_title is not None:
        plt.title(figure_title)


    transformed_points, pca = transform_PCA(points, components = 2)

    if cluster_predictions is not None:

        # Count number of clusters
        found_cluster_labels = []
        n_clusters = 0
        for pred in cluster_predictions:
            if pred not in found_cluster_labels:
                n_clusters+=1
                found_cluster_labels.append(pred)

        # Reorder the labels to have colors matching the cluster results
        #cluster_labels = np.choose(cluster_predictions, list(range(0, n_clusters))).astype(float)
    else:
        found_cluster_labels = [0]
        n_clusters = 1

    if cluster_color_palette is None:
        cluster_color_palette = centroid_color_palette = plt.cm.viridis(np.linspace(0, 1, n_clusters),alpha=0.8) #Equally spaced color 

    if cluster_predictions is not None:
        cluster_colors = [cluster_color_palette[i] for i in cluster_predictions]
    else:
        cluster_colors = [(0.0, 0.0, 0.0, 1.0)] * len(transformed_points)

    #print(cluster_colors)
    
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], 
                c=cluster_colors, 
                cmap=plt.cm.nipy_spectral,
                marker = ".",
                s=2
                #edgecolor='k'
                )

    if cluster_centroids is not None:
        cluster_centroids = pca.transform(cluster_centroids)

        assert centroid_color_palette is not None, "When providing a cluster_color_palette also provide a centroid_color_palette"
        centroid_colors = [centroid_color_palette[i] for i in range(len(cluster_centroids))]
        
        plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1],
                    c=centroid_colors,
                    cmap=plt.cm.nipy_spectral,
                    s=150,
                    linewidths = 3,
                    zorder = 10,
                edgecolor='k')

    fig = plt.gca()
    fig.axes.get_xaxis().set_ticks([])
    fig.axes.get_yaxis().set_ticks([])

    if scale_range is not None:
        plt.xlim(-scale_range[0], scale_range[0])
        plt.ylim(-scale_range[1], scale_range[1])

    if show:
        plt.show()

    if save_to_dir is not None:
        plt.savefig(os.path.join(save_to_dir, file_name), dpi = 300)

def visualize_3D(points, show = False, save_to_dir = None, 
    cluster_predictions = None, cluster_centroids = None, 
    file_name = "figure", figure_title = None, figure_caption = None,
    scale_range = None, cluster_color_palette = None, centroid_color_palette = None):

    assert len(scale_range) >= 3, "Wrong scale_range length: "+str(len(scale_range)) + " (should be 3 or more)"

    #3D visualization
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[.01, .01, .95, .85], elev=48, azim=134)

    plt.cla()


    #Set figure font
    plt.rcParams.update({'font.size': 7})
    #Set figure caption
    if figure_caption is not None:
        fig.text(.2, .05, figure_caption, ha='center')
    

    #Set figure font
    plt.rcParams.update({'font.size': 5})
    #Set title
    if figure_title is not None:
        plt.title(figure_title)

    transformed_points, pca = transform_PCA(points, components = 3)
    
    if cluster_predictions is not None:

        # Count number of clusters
        found_cluster_labels = []
        n_clusters = 0
        for pred in cluster_predictions:
            if pred not in found_cluster_labels:
                n_clusters+=1
                found_cluster_labels.append(pred)

        # Reorder the labels to have colors matching the cluster results
        #cluster_labels = np.choose(cluster_predictions, list(range(0, n_clusters))).astype(float)
    else:
        found_cluster_labels = [0]
        n_clusters = 1

    if cluster_color_palette is None:
        cluster_color_palette, centroid_color_palette =plt.cm.viridis(np.linspace(0, 1, n_clusters),alpha=0.8) #Equally spaced color 
    
    if cluster_predictions is not None:
        cluster_colors = [cluster_color_palette[i] for i in cluster_predictions]
    else:
        cluster_colors = [(0.0, 0.0, 0.0, 1.0)] * len(transformed_points)
    #print(cluster_colors)
    
    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], points[:, 2], 
                c=cluster_colors, 
                cmap=plt.cm.nipy_spectral,
                marker = ".",
                s=2#,
                #edgecolor='k'
                )

    if cluster_centroids is not None:
        cluster_centroids = pca.transform(cluster_centroids)
        
        assert centroid_color_palette is not None, "When providing a cluster_color_palette also provide a centroid_color_palette"
        centroid_colors = [centroid_color_palette[i] for i in range(len(cluster_centroids))]
        
        ax.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], cluster_centroids[:, 2],
                    c=centroid_colors,
                    cmap=plt.cm.nipy_spectral,
                    s=150,
                    linewidths = 3,
                    zorder = 10,
                edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    if scale_range is not None:
        plt.xlim(-scale_range[0], scale_range[0])
        plt.ylim(-scale_range[1], scale_range[1])
        ax.set_zlim(-scale_range[2], scale_range[2])

    if show:
        plt.show()

    if save_to_dir is not None:
        plt.savefig(os.path.join(save_to_dir, file_name), dpi = 300)

class VisualizationMethod:
    @abc.abstractmethod
    def __call__(self):
        pass

class ClusterVisualizer(VisualizationMethod):
    def __init__(self, clusterer, show = True, save_snapshot_images_to = None, smaller_scale_range = None, larger_scale_range = None, file_name_prefix = "_"):
        self.clusterer = clusterer
        self.smaller_scale_range = smaller_scale_range
        self.larger_scale_range = larger_scale_range
        self.save_snapshot_images_to = save_snapshot_images_to
        self.show = show
        self.file_name_prefix = file_name_prefix
        if os.path.exists(save_snapshot_images_to) and save_snapshot_images_to.startswith("snapshots"):
            shutil.rmtree(save_snapshot_images_to)
            os.makedirs(save_snapshot_images_to)
        else:
            os.makedirs(save_snapshot_images_to)
        self.cluster_color_palette, self.centroid_color_palette = generate_cluster_color_palette(self.clusterer.clusters)
    
    def generate_gif(self, duration = None, file_name_prefix = ""):
        images = []
        for filename in natsort.natsorted(os.listdir(self.save_snapshot_images_to)):
            if not filename.endswith(".png") or filename.endswith("original.png") \
                 or (file_name_prefix is not None and not filename.startswith(file_name_prefix)) \
                 or not os.path.isfile(os.path.join(self.save_snapshot_images_to,filename)):
                continue
            image = imageio.imread(os.path.join(self.save_snapshot_images_to, filename))
            #if cropX is not None:
            #    image = image[cropX[0]:cropX[1],:,:]
            #if cropY is not None:
            #    image = image[:,cropY[0]:cropY[1],:]
            images.append(image)
        if len(images) == 0:
            return
        
        if duration is None:
            frame_duration = 1.0
        else:
            frame_duration = duration/len(images)
        
        imageio.mimsave(os.path.join(self.save_snapshot_images_to, file_name_prefix+"animated.gif"), images, duration = frame_duration)

    def generate_final_comparison(self, file_name_prefix = ""):
        original_image_filename = os.path.join(self.save_snapshot_images_to, file_name_prefix + "original.png")
        print(original_image_filename)
        assert os.path.exists(original_image_filename)
        
        original_image = imageio.imread(original_image_filename)
        #original_image = original_image[10:-90,:,:]
        #original_image = original_image[:,100:-100,:]

        image_filenames = []
        for filename in natsort.natsorted(os.listdir(self.save_snapshot_images_to)):
            if not filename.endswith(".png") or filename.endswith("original.png") or filename.endswith("animated.gif") or filename.endswith("final_result_comparison.png")\
                 or (file_name_prefix is not None and not filename.startswith(file_name_prefix)) \
                 or not os.path.isfile(os.path.join(self.save_snapshot_images_to,filename)):
                continue
            image_filenames.append(filename)

        assert len(image_filenames) > 0, "No per-iteration images found"

        final_result_image = imageio.imread(os.path.join(self.save_snapshot_images_to, image_filenames[-1]))
        #final_result_image = final_result_image[10:-90,:,:]
        #final_result_image = final_result_image[:,100:-100,:]

        plt.clf()
        plt.cla()

        comparison_image = plt.figure(figsize = (1,2), dpi = 1000)
        comparison_image.tight_layout()
        plt.margins(0)
        
        
        gs = gridspec.GridSpec(1, 2)
        gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

        ax1 = plt.subplot(gs[0])      
        ax1.imshow(original_image)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_aspect('equal')
        ax1.margins(0)

        ax2 = plt.subplot(gs[1]) 
        ax2.imshow(final_result_image)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_aspect('equal')
        ax2.margins(0)
        
        #plt.show()
        plt.savefig(os.path.join(self.save_snapshot_images_to, file_name_prefix+"final_result_comparison.png"), bbox_inches='tight', pad_inches=0)
        
        plt.clf()
        plt.cla()
        
        return comparison_image

class Cluster3DVisualizer(ClusterVisualizer):
    def __call__(self, individual, iteration, file_name_prefix = "", figure_title = None, figure_caption = None, show_centroids = False, show_original_points = False, view_clusters = True):          
        #Apply clustering and extract centroids
        points, cluster_predictions, cluster_centroids = self.clusterer(individual, return_transformed_points = not show_original_points)

        if not show_centroids:
            cluster_centroids = None

        if not view_clusters:
            cluster_predictions = None

        if show_original_points:
            scale_range = self.smaller_scale_range
        else:
            scale_range = self.larger_scale_range

        #3D visualization as a scatter plot using PCA dimensionality reduction
        visualize_3D(points, 
                    show=self.show, 
                    cluster_predictions=cluster_predictions, cluster_centroids=cluster_centroids, 
                    save_to_dir = self.save_snapshot_images_to, file_name=file_name_prefix+"3d_" + str(iteration) + ".png", figure_title = figure_title, figure_caption = figure_caption,
                    scale_range = scale_range, cluster_color_palette = self.cluster_color_palette, centroid_color_palette = self.centroid_color_palette)

    def generate_gif(self, duration = None):
        super().generate_gif(duration = duration , file_name_prefix="original_3d_")
        super().generate_gif(duration = duration , file_name_prefix="transformed_3d_")

    def generate_final_comparison(self):
        super().generate_final_comparison(file_name_prefix="original_3d_")
        super().generate_final_comparison(file_name_prefix="transformed_3d_")

class Cluster2DVisualizer(ClusterVisualizer):
    def __call__(self, individual, iteration, file_name_prefix = "", figure_title = None, figure_caption = None, show_centroids = False, show_original_points = True, view_clusters = True):        
        #Apply clustering and extract centroids
        points, cluster_predictions, cluster_centroids = self.clusterer(individual, return_transformed_points = not show_original_points)

        if not show_centroids:
            cluster_centroids = None

        if not view_clusters:
            cluster_predictions = None

        if show_original_points:
            scale_range = self.smaller_scale_range
        else:
            scale_range = self.larger_scale_range

        #2D visualization as a scatter plot using PCA dimensionality reduction
        visualize_2D(points, 
                    show=self.show, 
                    cluster_predictions=cluster_predictions, cluster_centroids=cluster_centroids, 
                    save_to_dir = self.save_snapshot_images_to, file_name=file_name_prefix+"2d_" + str(iteration) + ".png", figure_title=figure_title, figure_caption = figure_caption,
                    scale_range = scale_range, cluster_color_palette = self.cluster_color_palette, centroid_color_palette = self.centroid_color_palette)

    def generate_gif(self, duration = None):
        super().generate_gif(duration = duration , file_name_prefix="original_2d_")
        super().generate_gif(duration = duration , file_name_prefix="transformed_2d_")

    def generate_final_comparison(self):
        super().generate_final_comparison(file_name_prefix="2d_")
        super().generate_final_comparison(file_name_prefix="transformed_2d_")

class Cluster2D3DVisualizer(ClusterVisualizer):
    def __call__(self, individual, iteration, file_name_prefix = "", figure_title = None, figure_caption = None, show_centroids = False, ground_truth = None, show_original_points = False, view_clusters = True):
        points, cluster_predictions, cluster_centroids = self.clusterer(individual, return_transformed_points = not show_original_points, ground_truth = ground_truth)

        if not show_centroids:
            cluster_centroids = None

        if not view_clusters:
            cluster_predictions = None

        if show_original_points:
            scale_range = self.smaller_scale_range
        else:
            scale_range = self.larger_scale_range

        #2D visualization as a scatter plot using PCA dimensionality reduction
        visualize_2D(points, 
                    show=self.show, 
                    cluster_predictions=cluster_predictions, cluster_centroids=cluster_centroids, 
                    save_to_dir = self.save_snapshot_images_to, file_name=file_name_prefix+"2d_" + str(iteration) + ".png", figure_title = figure_title, figure_caption = figure_caption,
                    scale_range = scale_range, cluster_color_palette = self.cluster_color_palette, centroid_color_palette = self.centroid_color_palette)

        #3D visualization as a scatter plot using PCA dimensionality reduction
        visualize_3D(points, 
                    show=self.show, 
                    cluster_predictions=cluster_predictions, cluster_centroids=cluster_centroids, 
                    save_to_dir = self.save_snapshot_images_to, file_name=file_name_prefix+"3d_" + str(iteration) + ".png", figure_title = figure_title, figure_caption = figure_caption,
                    scale_range = scale_range, cluster_color_palette = self.cluster_color_palette, centroid_color_palette = self.centroid_color_palette)

    def generate_gif(self, duration = None):
        super().generate_gif(duration = duration , file_name_prefix="original_2d_")
        super().generate_gif(duration = duration, file_name_prefix="original_3d_")
        super().generate_gif(duration = duration , file_name_prefix="transformed_2d_")
        super().generate_gif(duration = duration, file_name_prefix="transformed_3d_")

    def generate_final_comparison(self):
        super().generate_final_comparison(file_name_prefix="2d_")
        super().generate_final_comparison(file_name_prefix="3d_")
        #super().generate_final_comparison(file_name_prefix="transformed_2d_")
        #super().generate_final_comparison(file_name_prefix="transformed_3d_")


class Print(VisualizationMethod):        
    def __call__(self, individual):    
        print(individual)

if __name__ == "__main__":
    from optimization import GeneticBinaryOptimizer 
    genetic_learner = GeneticBinaryOptimizer.load("snapshots/SPECTRAL_4_clusters_Couples_Single_scales_MinimumPointDistance")
    genetic_learner.result_visualization.generate_final_comparison()
