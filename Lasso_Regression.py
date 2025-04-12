import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, n_iterations=1000, tol=1e-4):
        self.alpha = alpha                  # Regularization strength
        self.n_iterations = n_iterations    # Number of iterations for optimization
        self.tol = tol                      # Tolerance for convergence
        self.coefficients = None
        self.intercept = 0.0

    def _soft_threshold(self, value, threshold):
        """
        Soft thresholding function used in coordinate descent
        """
        if value > threshold:
            return value - threshold
        elif value < -threshold:
            return value + threshold
        else:
            return 0.0

    def fit(self, X, y):
        """
        Fit the Lasso model to data using coordinate descent.
        """
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = np.mean(y)
        y = y - self.intercept  # Center the target

        for iteration in range(self.n_iterations):
            coef_prev = self.coefficients.copy()

            for j in range(n_features):
                # Partial residual (excluding feature j)
                residual = y - X @ self.coefficients + self.coefficients[j] * X[:, j]

                # Compute rho
                rho = np.dot(X[:, j], residual)

                # Update coefficient j
                self.coefficients[j] = self._soft_threshold(rho, self.alpha) / (np.dot(X[:, j], X[:, j]) + 1e-8)

            # Check for convergence
            if np.max(np.abs(self.coefficients - coef_prev)) < self.tol:
                break

    def predict(self, X):
        """
        Predict values using the learned model.
        """
        return X @ self.coefficients + self.intercept

if __name__ == "__main__":
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.2, 1.9, 3.2, 3.8, 5.5])

    # Initialize and fit the model
    lasso = LassoRegression(alpha=0.5)
    lasso.fit(X, y)

    # Predict
    preds = lasso.predict(np.array([[6], [7]]))
    print("Predictions:", preds)
