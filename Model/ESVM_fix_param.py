import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class ESVM:
    def __init__(self):
        self.best_model = None

    def build(self):
        # Cấu hình cố định
        config = {
            'pca__n_components': min(35, self.X_train.shape[0], self.X_train.shape[1]) ,
            'svc__C': 0.6295660257538724,
            'svc__kernel': 'rbf',
            'svc__gamma': 0.09798977145917381,
            'adaboost__n_estimators': 87,
            'adaboost__learning_rate': 0.40967575300614606
        }

        pipeline = Pipeline([
            ('pca', PCA(n_components=config['pca__n_components'])),
            ('adaboost', AdaBoostClassifier(
                base_estimator=SVC(probability=True, kernel=config['svc__kernel'], C=config['svc__C'], gamma=config['svc__gamma'], random_state=42),
                n_estimators=config['adaboost__n_estimators'],
                learning_rate=config['adaboost__learning_rate'],
                random_state=42
            ))
        ])
        return pipeline

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.best_model = self.build()
        self.best_model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.best_model is not None:
            return np.round(self.best_model.predict(X_test))
        else:
            raise ValueError("Model is not trained yet. Call fit() first.")

    def predict_proba(self, X_test):
        if self.best_model is not None:
            return self.best_model.predict_proba(X_test)
        else:
            raise ValueError("Model is not trained yet. Call fit() first.")
