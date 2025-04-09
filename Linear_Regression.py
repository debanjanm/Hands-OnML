import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None  # weights
        self.intercept = None     # bias

    def fit(self, X, y):
        """
        Fit the linear model to the training data.
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        """
        # Add bias column (intercept) to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of 1s

        # Normal Equation: (X^T X)^-1 X^T y
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept = theta_best[0]
        self.coefficients = theta_best[1:]

    def predict(self, X):
        """
        Predict using the linear model.
        X: np.ndarray of shape (n_samples, n_features)
        Returns: np.ndarray of shape (n_samples,)
        """
        return X @ self.coefficients + self.intercept
if __name__ == "__main__":
    # Example usage
    import numpy as np
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict
    predictions = model.predict(np.array([[6], [7]]))
    print("Predictions:", predictions)
