from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd

# Cố định seed
def set_seed(seed=42):
    np.random.seed(seed)

# Gọi hàm set_seed() trong mã của bạn
set_seed(42)

class WeightedRegression:
    def __init__(self, base_model=None, weight=0.5):
        """
        Weighted Regression combining global and individualized models.

        Parameters:
        - base_model: Base regression model (default: LinearRegression).
        - weight: Weight for individualized model predictions (0 <= weight <= 1).
        """
        self.base_model = base_model if base_model else LinearRegression()
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
        Predict using weighted combination of global and individualized models.

        Parameters:
        - X: Feature matrix (2D array-like).
        - group_ids: Group IDs for individualized models (1D array-like).

        Returns:
        - Weighted predictions (1D array).
        """
        y_global_pred = self.global_model.predict(X)
        y_individual_pred = np.zeros_like(y_global_pred)

        for i, group in enumerate(group_ids):
            if group in self.individual_models:
                X_row = X.iloc[i].values.reshape(1, -1) if isinstance(X, pd.DataFrame) else X[i].reshape(1, -1)
                y_individual_pred[i] = self.individual_models[group].predict(X_row)
            else:
                y_individual_pred[i] = y_global_pred[i]

        return self.weight * y_individual_pred + (1 - self.weight) * y_global_pred

    def optimize_weight(self, X, y, group_ids):
        """
        Optimize weight to minimize mean squared error on given data.

        Parameters:
        - X: Feature matrix (2D array-like).
        - y: Target vector (1D array-like).
        - group_ids: Group IDs for individualized models (1D array-like).

        Returns:
        - Best weight value (float).
        """
        def weighted_mse(weight):
            self.weight = weight
            y_pred = self.predict(X, group_ids)
            return mean_squared_error(y, y_pred)

        from scipy.optimize import minimize
        result = minimize(weighted_mse, x0=0.5, bounds=[(0, 1)])
        self.weight = result.x[0]
        self.best_params = self.weight
    
    def predict(self, y_true, y_prob):
        thresholds = np.linspace(min(y_prob), max(y_prob), 100)  # Thử nghiệm 100 ngưỡng
        best_accuracy = 0
        best_predict = None

        for threshold in thresholds:
            predicted_classes = (y_prob >= threshold).astype(int)
            accuracy = accuracy_score(y_true, predicted_classes)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_predict = predicted_classes

        return best_predict
