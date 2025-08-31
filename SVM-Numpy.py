import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization strength
        self.n_iters = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train using gradient descent on hinge loss.
        y should be -1 or 1
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert y from 0/1 to -1/1 if needed
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = -y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, 0)

if __name__ == "__main__":
    # Training data
    X = np.array([[1, 2], [2, 3], [3, 3.5], [6, 7], [7, 8]])
    y = np.array([0, 0, 0, 1, 1])

    # Train the model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    svm.fit(X, y)

    # Predict
    predictions = svm.predict(np.array([[3, 4], [6, 6]]))
    print("Predictions:", predictions)
