import numpy as np

class KMeans:
    """
    KMeans clustering algorithm implementation.

    Parameters:
    k (int): Number of clusters.
    max_iter (int): Maximum number of iterations.
    """

    def __init__(self, k, max_iter):
        """
        Initialize KMeans object with specified number of clusters and maximum iterations.

        Parameters:
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        """
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.X_train = None
        self.clusters = {}  # Initialize clusters as empty dictionary
        for j in range(self.k):
            self.clusters[j] = []

    def initialize(self, centroids):
        """
        Initialize centroids with given values.

        Parameters:
        centroids (dict): Initial centroid values.
        """
        self.centroids = {}  # Initialize centroids as empty dictionary
        for i in range(self.k):
            self.centroids[i] = centroids[i]

    def wcss(self):
        """
        Compute the Within-Cluster Sum of Squares (WCSS).

        Returns:
        float: WCSS value.
        """
        sum = 0  # Initialize sum
        list_centroids = list(self.centroids.values())
        for vector in self.X_train:
            # Find the closest centroid for each data point and sum the squared distances
            icentroid = KMeans.close_centroid(vector, list_centroids)
            sum += KMeans.distance(list_centroids[icentroid], vector) ** 2
        return sum

    def fit(self, X_train):
        """
        Fit the KMeans model to the training data.

        Parameters:
        X_train (numpy.ndarray): Training data.

        Returns:
        dict: Final centroids.
        """
        self.X_train = X_train
        for i in range(self.max_iter):
            self.clusters = {}  # Reset clusters
            for j in range(self.k):
                self.clusters[j] = []

            # Assign each point to the nearest centroid
            for p in X_train:
                distances = []
                for j in self.centroids:
                    distances.append(self.distance(p, self.centroids[j]))
                icluster = distances.index(min(distances))
                self.clusters[icluster].append(p)

            end = dict(self.centroids)

            # Update centroids to the mean of their assigned points
            for icluster in self.clusters:
                if self.clusters[icluster]:  # Avoid division by zero
                    self.centroids[icluster] = np.average(self.clusters[icluster], axis=0)

            # Check for convergence (if centroids do not change)
            converged = True
            for c in self.centroids:
                if not np.allclose(self.centroids[c], end[c]):
                    converged = False
                    break
            if converged:
                break
        return self.centroids

    def predict(self, X):
        """
        Predict the cluster labels for the given data.

        Parameters:
        X (numpy.ndarray): Data to predict labels for.

        Returns:
        numpy.ndarray: Predicted labels.
        """
        labels = []
        for vector in X:
            labels.append(self.predict_for_vector(vector))
        return np.array(labels)

    def predict_for_vector(self, V):
        """
        Predict the cluster label for a single data vector.

        Parameters:
        V (numpy.ndarray): Data vector.

        Returns:
        int: Predicted cluster label.
        """
        distances = [KMeans.distance(V, self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    @staticmethod
    def close_centroid(vector, centroids):
        """
        Find the index of the closest centroid to the given vector.

        Parameters:
        vector (numpy.ndarray): Data vector.
        centroids (list): List of centroid vectors.

        Returns:
        int: Index of the closest centroid.
        """
        distances = [KMeans.distance(vector, p) for p in centroids]
        return np.argmin(distances)

    @staticmethod
    def distance(v1, v2):
        """
        Compute the Euclidean distance between two vectors.

        Parameters:
        v1 (numpy.ndarray): First vector.
        v2 (numpy.ndarray): Second vector.

        Returns:
        float: Euclidean distance between v1 and v2.
        """
        return np.sqrt(np.sum((v1 - v2) ** 2))
