import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # regularization strength
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit ridge regression model.
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Add bias term
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Identity matrix (with 0 for intercept term)
        I = np.eye(n_features + 1)
        I[0, 0] = 0  # don't regularize the bias

        # Closed-form solution (normal equation with regularization)
        A = X_b.T @ X_b + self.alpha * I
        b = X_b.T @ y
        theta = np.linalg.inv(A) @ b

        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def predict(self, X):
        """
        Predict using the ridge regression model.
        """
        return X @ self.coefficients + self.intercept

if __name__ == "__main__":
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 2, 3, 4, 6])

    # Train Ridge Regression
    ridge = RidgeRegression(alpha=0.5)
    ridge.fit(X, y)

    # Predict
    preds = ridge.predict(np.array([[6], [7]]))
    print("Predictions:", preds)
