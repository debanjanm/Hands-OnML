import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # convergence tolerance
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Perform k-means clustering.
        X: np.ndarray of shape (n_samples, n_features)
        """
        n_samples, _ = X.shape

        # Randomly choose k unique data points as initial centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign clusters based on closest centroid
            self.labels = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])

            # Check for convergence
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        Assigns each point to the nearest centroid.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # shape (n_samples, k)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        """
        Predict the closest cluster for each point.
        """
        return self._assign_clusters(X)

if __name__ == "__main__":
    # Sample data
    X = np.array([
        [1, 2], [1.5, 1.8], [5, 8],
        [8, 8], [1, 0.6], [9, 11]
    ])

    # Run k-means
    kmeans = KMeans(k=2)
    kmeans.fit(X)

    print("Centroids:\n", kmeans.centroids)
    print("Labels:", kmeans.labels)

    # Predict for new data points
    new_points = np.array([[0, 0], [10, 10]])
    predictions = kmeans.predict(new_points)
    print("Cluster predictions for new points:", predictions)
