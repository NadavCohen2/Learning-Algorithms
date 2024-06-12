import numpy as np


class KNN:
    def __init__(self, k) -> None:
        """
        Initialize the K-Nearest Neighbors classifier.

        Parameters:
        k (int): The number of nearest neighbors to consider.
        """
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train) -> None:
        """
        Fit the model using the given training data.

        Parameters:
        x_train (numpy.ndarray): Training samples, a 2D array where each row is a sample.
        y_train (numpy.ndarray): Training labels, a 1D array where each element is a label for the corresponding sample.
        """
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """
        Predict the labels for the given test data.

        Parameters:
        x_test (numpy.ndarray): Test samples, a 2D array where each row is a sample.

        Returns:
        numpy.ndarray: Predicted labels, a 1D array where each element is the predicted label for the corresponding sample.
        """
        labels = []
        for vector in x_test:
            # Predict the label for each test vector
            labels.append(self.vector_predict(vector))
        return np.array(labels)

    def vector_predict(self, row):
        """
        Predict the label for a single sample using the k-nearest neighbors algorithm.

        Parameters:
        row (numpy.ndarray): A single sample, a 1D array.

        Returns:
        int: The predicted label.
        """
        distance_array = []
        for i in self.x_train:
            # Calculate Euclidean distance from the test sample to each training sample
            distance_array.append(KNN.distance(row, i))

        # Get the indices of the k nearest neighbors
        k_indexes = np.argsort(distance_array)[:self.k]

        # Extract the labels of the k nearest neighbors
        k_labels = [self.y_train[j] for j in k_indexes]

        # Return the most common label among the k nearest neighbors
        return np.bincount(k_labels).argmax()

    @staticmethod
    def distance(v1, v2):
        """
        Compute the Euclidean distance between two vectors.

        Parameters:
        v1 (numpy.ndarray): First vector.
        v2 (numpy.ndarray): Second vector.

        Returns:
        float: The Euclidean distance between the two vectors.
        """
        return np.sqrt(np.sum((v1 - v2) ** 2))
