import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
import optuna
from sklearn.metrics import accuracy_score

class ESVM:
    def __init__(self):
        self.best_model = None
        self.best_params = None

    def build(self, config):
    # Tạo pipeline với PCA và AdaBoost
        pipeline = Pipeline([
            ('pca', PCA(n_components=config['pca__n_components'])),  # PCA
            ('adaboost', AdaBoostClassifier(
                base_estimator=SVC(probability=True, kernel=config['svc__kernel'], C=config['svc__C'], gamma=config['svc__gamma'], random_state=42),
                n_estimators=config['adaboost__n_estimators'],
                learning_rate=config['adaboost__learning_rate'],
                random_state=42
            ))
        ])
        return pipeline

    def training(self, trial, X_train, y_train):
        # Đề xuất siêu tham số cho SVC, PCA và AdaBoost
        config = {
          'pca__n_components': 35,
          'svc__C': 0.6295660257538724,
          'svc__kernel': 'rbf',
          'svc__gamma': 0.09798977145917381,
          'adaboost__n_estimators': 87,
          'adaboost__learning_rate': 0.40967575300614606
      }

        pipeline = self.build(config)
        pipeline.fit(X_train, y_train)
        
        # Đánh giá mô hình trên tập kiểm tra và trả lại độ chính xác để tối ưu hóa
        y_pred = pipeline.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        return accuracy

    def objective(self, trial, X_train, y_train):
        # Đối tượng tối ưu hóa
        return self.training(trial, X_train, y_train)

    def fit(self, X_train, y_train):
        # Tạo đối tượng study của Optuna
        study = optuna.create_study(direction='maximize')  # maximize để tối ưu hóa độ chính xác
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=50)

        # Lấy các tham số tốt nhất từ Optuna
        self.best_params = study.best_params
        self.best_model = self.build(self.best_params)
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
