import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predict the class labels for the given data.
        X: np.ndarray of shape (n_queries, n_features)
        Returns: np.ndarray of shape (n_queries,)
        """
        predictions = [self._predict_point(x) for x in X]
        return np.array(predictions)

    def _predict_point(self, x):
        # Compute distances to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get their labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    # Example data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]])
    y_train = np.array([0, 0, 0, 1, 1])

    X_test = np.array([[2, 2], [6, 6]])

    # Train and predict
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    print("Predictions:", predictions)
