from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Cố định seed
def set_seed(seed=42):
    np.random.seed(seed)

# Gọi hàm set_seed() trong mã của bạn
set_seed(42)

class WeightedLogisticRegression:
    def __init__(self, base_model=None, weight=0.5):
        """
        Weighted Logistic Regression combining global and individualized models.

        Parameters:
        - base_model: Base classification model (default: LogisticRegression).
        - weight: Weight for individualized model predictions (0 <= weight <= 1).
        """
        self.base_model = base_model if base_model else LogisticRegression()
        self.weight = weight
        self.global_model = None
        self.individual_models = {}
        self.best_params = None

    def fit(self, X, y, group_ids):
        """
        Fit global and individualized models.

        Parameters:
        - X: Feature matrix (2D array-like).
        - y: Target vector (1D array-like).
        - group_ids: Group IDs for individualized models (1D array-like).
        """
        self.global_model = self.base_model
        self.global_model.fit(X, y)

        unique_groups = np.unique(group_ids)
        for group in unique_groups:
            group_indices = [i for i, val in enumerate(group_ids) if val == group]
            X_group = X.iloc[group_indices]
            y_group = y[group_indices]

            model = self.base_model.__class__()
            model.fit(X_group, y_group)
            self.individual_models[group] = model

    def predict_proba(self, X, group_ids):
        """
        Predict probabilities using weighted combination of global and individualized models.

        Parameters:
        - X: Feature matrix (2D array-like).
        - group_ids: Group IDs for individualized models (1D array-like).

        Returns:
        - Weighted probabilities (1D array).
        """
        y_global_pred = self.global_model.predict_proba(X)
        y_individual_pred = np.zeros_like(y_global_pred)

        for i, group in enumerate(group_ids):
            if group in self.individual_models:
                X_row = X.iloc[i].values.reshape(1, -1) if isinstance(X, pd.DataFrame) else X[i].reshape(1, -1)
                y_individual_pred[i] = self.individual_models[group].predict_proba(X_row)[:, 1]
            else:
                y_individual_pred[i] = y_global_pred[i]

        return self.weight * y_individual_pred + (1 - self.weight) * y_global_pred

    def optimize_weight(self, X, y, group_ids):
        """
        Optimize weight to maximize accuracy on given data.

        Parameters:
        - X: Feature matrix (2D array-like).
        - y: Target vector (1D array-like).
        - group_ids: Group IDs for individualized models (1D array-like).

        Returns:
        - Best weight value (float).
        """
        def weighted_accuracy(weight):
            self.weight = weight
            y_prob = self.predict_proba(X, group_ids)
            thresholds = np.linspace(0, 1, 100)
            best_accuracy = 0
            
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
            
            return -best_accuracy  # Maximize accuracy

        from scipy.optimize import minimize
        result = minimize(weighted_accuracy, x0=0.5, bounds=[(0, 1)])
        self.weight = result.x[0]
        self.best_params = self.weight
    
    def predict(self, X, group_ids):
        """
        Predict class labels using the best threshold.

        Parameters:
        - X: Feature matrix (2D array-like).
        - group_ids: Group IDs for individualized models (1D array-like).

        Returns:
        - Predicted class labels (1D array).
        """
        y_prob = self.predict_proba(X, group_ids)
        thresholds = np.linspace(0, 1, 100)
        best_accuracy = 0
        best_predict = None

        for threshold in thresholds:
            predicted_classes = (y_prob >= threshold).astype(int)
            accuracy = accuracy_score(y, predicted_classes)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_predict = predicted_classes

        return best_predict
