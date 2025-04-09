import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predict probability estimates.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Sample data
    # Features (X) and labels (y)
    # Binary classification example
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Predictions
    print("Probabilities:", model.predict_proba(np.array([[1.5], [3.5]])))
    print("Predicted classes:", model.predict(np.array([[1.5], [3.5]])))
