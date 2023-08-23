import numpy as np


class KNN:
    def __init__(self, k) -> None:
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        labels = []
        for vector in x_test: #מעבר לפי וקטורים וחישוב התווית שלהם
            labels.append(self.vector_predict(vector))
        return np.array(labels)

    def vector_predict(self, row):
        distance_array = []
        for i in self.x_train: #חישוב המרחקים מכל הדוגמאות
            distance_array.append(KNN.distance(row, i))
        k_indexes = np.argsort(distance_array)[0:self.k] #לקיחת הכי קרובים לפי K
        k_labels = []
        for j in k_indexes: #התאמת התווית לוקטור
            k_labels.append(self.y_train[j])
        k_labels = np.array(k_labels)
        return np.bincount(k_labels).argmax()

    @staticmethod
    def distance(v1, v2):
        return np.sqrt((np.sum((v1 - v2) ** 2))) #חישוב מרחק בין 2 וקטורים