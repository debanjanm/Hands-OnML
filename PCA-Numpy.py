import numpy as np

class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        # Step 1: center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: covariance matrix
        cov_mat = np.cov(X_centered, rowvar=False)

        # Step 3: eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

        # Step 4: sort eigenvalues and eigenvectors
        sorted_idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[sorted_idx]
        eigen_vecs = eigen_vecs[:, sorted_idx]

        # select top components
        self.components = eigen_vecs[:, :self.n_components]
        self.explained_variance = eigen_vals[:self.n_components] / np.sum(eigen_vals)

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    # create dummy data: 100 samples, 5 features
    np.random.seed(42)
    X_dummy = np.random.rand(100, 5) * 100

    pca = MyPCA(n_components=2)
    X_proj = pca.fit_transform(X_dummy)

    print("Projected shape:", X_proj.shape)
    print("Explained variance ratios:", pca.explained_variance)
