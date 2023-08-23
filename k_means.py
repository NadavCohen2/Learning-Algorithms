import numpy as np


class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.X_train = None
        self.clusters = {}
        for j in range(self.k):
            self.clusters[j] = []

    def initialize(self, centroids):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = centroids[i]

    def wcss(self):
        sum = 0
        list_centroids = list(self.centroids.values())
        for vector in self.X_train:
            icentroid = KMeans.close_centroid(vector, list_centroids)  # מציאת המרכז הכי קרוב
            sum += KMeans.distance(list_centroids[icentroid], vector) ** 2  # חיבור המרכזים הכי קרובים
        return sum

    def fit(self, X_train):
        self.X_train = X_train
        for i in range(self.max_iter):
            for p in X_train: #עובר על כל וקטור
                distances = []
                for j in self.centroids:
                    distances.append(self.distance(p, self.centroids[j]))
                icluster = distances.index(min(distances)) # מוצא את המרכז הכי קרוב
                self.clusters[icluster].append(p)
            end = dict(self.centroids)
            for icluster in self.clusters:
                self.centroids[icluster] = np.average(self.clusters[icluster], axis=0)  # מוצא את המרכז האידאלי
            for c in self.centroids:
                original_centroid = end[c]
                current_centroid = self.centroids[c]
                if np.all(current_centroid == original_centroid):  # אם לא נעשה שינוי באיטרציה האחרונה הגענו לפתרון
                    break
        return self.centroids

    def predict(self, X):
        labels = []
        for vector in X:
            labels.append(self.predict_for_vector(vector))
        return np.array(labels)

    def predict_for_vector(self, V):
        distances = [KMeans.distance(V, self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    @staticmethod
    def close_centroid(vector, centroids):  # מציאת המרכז הכי קרוב לוקטור
        distances = [KMeans.distance(vector, p) for p in centroids]
        return np.argmin(distances)

    @staticmethod
    def distance(v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))  # חישוב מרחק בין 2 וקטורים
