'''kmeans.py
Performs K-Means clustering
YOUR NAME HERE
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
from palettable.colorbrewer.qualitative import Set2_6


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        difference = pt_2-pt_1
        return np.sqrt(np.dot(difference, difference))
    
    def pt_to_pt_distance_matrix(self):
        D = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[1]): # for all the columns in data
            D += (np.atleast_2d(self.data[:, i]) - np.atleast_2d(self.data[:, i]).T) ** 2
        return np.sqrt(D)
    
    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        difference = pt - centroids
        return np.sqrt(np.diagonal(np.dot(difference, difference.T)))

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''

        
        centroidIds = np.random.choice(
            np.arange(len(self.data)), size=k, replace=False)
        # centroids = np.array([self.data[x] for x in centroidIds])
        centroids = self.data[centroidIds,:]
        self.k = k

        return centroids


    def cluster(self, k=2, tol=1e-5, max_iter=1000, verbose=False, silouette=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''

        centroids = self.initialize(k)
        itr = 0
        diff = np.ones(k*self.num_features).reshape(k, self.num_features) + tol
        while itr < max_iter and (diff > tol).any():
            data_centroid_labels = self.update_labels(centroids)
            centroids, diff = self.update_centroids(
                k, data_centroid_labels, centroids)
            self.data_centroid_labels = data_centroid_labels
            self.centroids = centroids
            if silouette == True:
                self.inertia = self.silhouette()  # s value
            else:
                self.inertia = self.compute_inertia()
            itr += 1

        return self.inertia, itr


    def cluster_batch(self, k=2, n_iter=1, verbose=False, silouette=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''

        best_centroids = self.centroids
        best_data_centroid_labels = self.data_centroid_labels
        best_inertia = 10e9

        if silouette == True:
            best_inertia = -2

        for i in range(n_iter):
            inertia, n = self.cluster(k, silouette=silouette)
            if silouette == False:
                if inertia < best_inertia:
                    best_inertia = self.inertia
                    best_data_centroid_labels = self.data_centroid_labels
                    best_centroids = self.centroids
            else:
                if inertia > best_inertia:
                    best_inertia = self.inertia
                    best_data_centroid_labels = self.data_centroid_labels
                    best_centroids = self.centroids

        self.inertia = best_inertia
        self.data_centroid_labels = best_data_centroid_labels
        self.centroids = best_centroids



    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        
        '''
        if np.any(data_centroid_labels == (k-1)):
                    # havbe something in the cluster
                    # do the jth centroid
                    #print("normal")
                    prev_centroids = prev_centroids
                    new_centroids = np.array([np.mean(np.array([self.data[i] for i in range(len(
                        data_centroid_labels)) if data_centroid_labels[i] == j]), axis=0) for j in range(k)])
                    
        else:
                    #print("random")
                    #choose a random point and put it into the jth centroid
                    centroidIds = np.random.choice(np.arange(len(self.data)), size=k, replace=False)
                    new_centroids = np.array([self.data[x] for x in centroidIds])
        '''

        new_centroids = np.zeros((k, self.num_features))
        for j in range(k):
            prev_centroids = prev_centroids
            if np.any(data_centroid_labels == j):
                    # have something in the cluster
                    # do the jth centroid
                    new_centroids[j] = np.mean(self.data[data_centroid_labels == j], axis = 0)
                                       
                        
            else:
                #choose a random point and put it into the jth centroid
                centroidIds = np.random.choice(np.arange(len(self.data)))
                new_centroids[j] = self.data[centroidIds]
        

        
        
        # if  np.any(data_centroid_labels != k):
        #     #print(data_centroid_labels, "k: ", k)
        #     centroidIds = np.random.choice(np.arange(len(self.data)), size=k, replace=False)
        #     new_centroids = np.array([self.data[x] for x in centroidIds])
        # else:
        #     prev_centroids = prev_centroids
        #     new_centroids = np.array([np.mean(np.array([self.data[i] for i in range(len(
        #     data_centroid_labels)) if data_centroid_labels[i] == j]), axis=0) for j in range(k)])

        centroid_diff = new_centroids - prev_centroids
        self.k = k
        return new_centroids, centroid_diff

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = []
        for pt in self.data:
            dist = self.dist_pt_to_centroids(pt, centroids)
            labels.append(np.where(dist == np.min(dist))[0][0])

        labels = np.array(labels)
        return labels


    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''

        distance = 0

        for i in range(self.data.shape[0]):
            distance += (self.dist_pt_to_pt(self.data[i],
                        self.centroids[self.data_centroid_labels[i]]))**2
        inertia = distance / self.data.shape[0]

        return inertia


    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''

        cmap = Set2_6.mpl_colormap
        plt.scatter(self.data[:, 0], self.data[:, 1],
                    c=self.data_centroid_labels, cmap=cmap)
        plt.scatter(self.centroids[:, 0],
                    self.centroids[:, 1], c='k', marker="^")


    def elbow_plot(self, max_k,n_iter=1, silouette = False):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertia_list = []

        for i in range(max_k):
            self.cluster_batch(i+1,n_iter,silouette = silouette)
            inertia_list.append(self.inertia)
        
        plt.plot(np.arange(max_k)+1,inertia_list)
        plt.xlabel('k clusters')
        if silouette == True:
            plt.ylabel('silouette coefficient')
        else:
            plt.ylabel('inertia')
        plt.xticks(np.arange(1, max_k+1, 1))


    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        self.data = np.array([self.centroids[i]
                            for i in self.data_centroid_labels])


    def silhouette(self):
        D = self.pt_to_pt_distance_matrix()
        s = 0

        for i in range(self.num_samps):

            a = self.compute_a(i, D)
            b = self.compute_b(i, D)

            s += (b - a)/max(a, b)

        return s/self.num_samps


    def compute_a(self, i, D):
        distance = 0
        n = 0
        label = self.data_centroid_labels[i]

        for j in range(self.num_samps):
            if self.data_centroid_labels[j] == label:
                distance += D[i, j]
                n += 1
        distance = (1/n)*distance
        return distance


    def compute_b(self, i, D):
        label_i = self.data_centroid_labels[i]
        # k number of 0s matrix, where k is the number of clusters
        distance = np.zeros(self.k)

        num_points_in_cluster = np.zeros(self.k)

        for j in range(self.num_samps):

            label_j = self.data_centroid_labels[j]

            if label_j != label_i:
                # should this part be distance[j]
                distance[label_j] = distance[label_j] + D[i, j]
                num_points_in_cluster[label_j] += 1

        num_points_in_cluster[label_i] = 1
        list_b = distance/num_points_in_cluster
        list_b[label_i] = 10e99
        min = list_b.min()

        return min

    # source https://jianjiesun.medium.com/dl-ml筆記-十-silhouette-coefficient-輪廓係數-2cfaa9e3a374
