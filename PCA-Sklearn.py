import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class SklearnPCAWrapper:
    def __init__(self, n_components=None, scale_data=True, **pca_kwargs):
        """
        n_components: number of principal components to keep (int, float or None for all).
        scale_data: whether to standardize features before PCA.
        pca_kwargs: extra arguments passed to sklearn.decomposition.PCA.
        """
        self.n_components = n_components
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.pca = PCA(n_components=n_components, **pca_kwargs)
        self.fitted = False

    def fit(self, X):
        """Fit the scaler (if enabled) and PCA on the data."""
        X_proc = self.scaler.fit_transform(X) if self.scale_data else X
        self.pca.fit(X_proc)
        self.fitted = True
        return self

    def transform(self, X):
        """Apply PCA transformation to new data."""
        if not self.fitted:
            raise ValueError("SklearnPCAWrapper instance is not fitted yet.")
        X_proc = self.scaler.transform(X) if self.scale_data else X
        return self.pca.transform(X_proc)

    def fit_transform(self, X):
        """Fit and immediately transform the data."""
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio_(self):
        """Proportion of explained variance per component."""
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        return self.pca.explained_variance_ratio_

    @property
    def components_(self):
        """Principal axes in feature space."""
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        return self.pca.components_

    @property
    def n_components_(self):
        """Actual components kept (especially when n_components=None or a float threshold)."""
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        return self.pca.n_components_

if __name__ == "__main__":
    # Generate dummy data
    np.random.seed(0)
    X_dummy = np.random.randn(200, 10) * 50 + 20

    # Create and use the wrapper
    pca_wrapper = SklearnPCAWrapper(n_components=0.9)  # Retain 90% variance
    X_reduced = pca_wrapper.fit_transform(X_dummy)

    print("Reduced shape:", X_reduced.shape)
    print("Explained variance ratio:", pca_wrapper.explained_variance_ratio_)
    print("Number of components retained:", pca_wrapper.n_components_)
