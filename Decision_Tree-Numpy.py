import numpy as np
import pandas as pd

class DecisionTreeClassifierScratch:

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _entropy(self, y):
        """Calculates entropy of a set of labels."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _gini(self, y):
        """Calculates Gini impurity of a set of labels."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _best_split(self, X, y, criterion='gini'):
        """Finds the best feature and split point to divide the data."""

        best_gain = -1  # Initialize with a value that will be improved
        best_feature = None
        best_threshold = None

        if criterion == 'gini':
            impurity_func = self._gini
        elif criterion == 'entropy':
            impurity_func = self._entropy
        else:
            raise ValueError("Invalid criterion. Choose 'gini' or 'entropy'.")

        parent_impurity = impurity_func(y)

        for feature in range(X.shape[1]):  # Iterate through features
            values = np.unique(X[:, feature])  # Unique values for the feature

            for threshold in values:  # Try each unique value as a threshold
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                y_left, y_right = y[left_indices], y[right_indices]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue  # Skip if a split creates a leaf smaller than minimum

                n = len(y)
                n_left = len(y_left)
                n_right = len(y_right)

                gain = parent_impurity - (n_left / n) * impurity_func(y_left) - (n_right / n) * impurity_func(y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively builds the decision tree."""
        if depth == self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))  # Leaf node: majority class

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None: # No good split found
            return np.argmax(np.bincount(y)) # Leaf node: majority class

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        """Fits the decision tree to the training data."""
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, tree):
        """Predicts the class for a single data point."""
        if isinstance(tree, (int, np.int64)):  # Leaf node
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

    def predict(self, X):
        """Predicts the classes for a set of data points."""
        return np.array([self._predict_one(x, self.tree) for x in X])


# Example usage:
X = np.array([[1, 2], [2, 2], [1, 3], [5, 4], [6, 4], [5, 5]])
y = np.array([0, 0, 0, 1, 1, 1])

clf = DecisionTreeClassifierScratch(max_depth=3, min_samples_split=2, min_samples_leaf=1)
clf.fit(X, y)

predictions = clf.predict(X)
print("Predictions:", predictions)

# Example with Pandas DataFrame
data = {'feature1': [1, 2, 1, 5, 6, 5], 'feature2': [2, 2, 3, 4, 4, 5], 'label': [0, 0, 0, 1, 1, 1]}
df = pd.DataFrame(data)
X_df = df[['feature1', 'feature2']].values  # Convert to NumPy array
y_df = df['label'].values

clf_df = DecisionTreeClassifierScratch(max_depth=3, min_samples_split=2, min_samples_leaf=1)
clf_df.fit(X_df, y_df)
predictions_df = clf_df.predict(X_df)
print("Predictions (DataFrame):", predictions_df)