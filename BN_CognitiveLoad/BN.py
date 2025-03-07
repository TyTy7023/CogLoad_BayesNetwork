import sys
import pandas as pd
import numpy as np

sys.path.append('/kaggle/working/cogload/')
from install_library import install_and_import
install_and_import('pgmpy')

sys.path.append('/kaggle/working/cogload/Exploratory_Data')
from EDA import EDA

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from sklearn.metrics import accuracy_score

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
from sklearn.model_selection import GroupKFold

class BayesianNetwork:
    def __init__(self, data, method='hill_climbing'):
        self.data = data
        self.target = 'Labels'
        self.method = method
        if method not in ['hill_climbing', 'tabu_search']:
            raise ValueError('Method not found')
        self.edges = self.Hill_climbing(data) if method == 'hill_climbing' else self.tabu_search(data)

    def Hill_climbing(self, data):
        hc = HillClimbSearch(data)
        hill_dag = hc.estimate(scoring_method=BicScore(data))
        return hill_dag.edges()
    
    def tabu_search(self, data, max_iter=100, tabu_length=10):
        tabu_list = []
        hc = HillClimbSearch(data)

        # Khởi tạo mô hình
        best_model = hc.estimate(scoring_method=BicScore(data))
        best_score = BicScore(data).score(best_model)

        for _ in range(max_iter):
            # Biến đổi mô hình hiện tại
            candidate = mutate_graph(best_model.copy())
            
            if any(set(candidate.edges) == set(tabu.edges) for tabu in tabu_list):
                continue  # Bỏ qua nếu đồ thị này đã trong danh sách Tabu

            score = BicScore(data).score(candidate)
            if score > best_score:  # Nếu mô hình mới tốt hơn, cập nhật
                best_model = candidate
                best_score = score

            # Thêm vào danh sách Tabu
            tabu_list.append(candidate)
            if len(tabu_list) > tabu_length:
                tabu_list.pop(0)  # Giữ danh sách Tabu có kích thước giới hạn

        return best_model.edges()
    
    def fit(self, X_train, y_train, user_train):
        '''
        Fit Bayesian Network
        '''
        
        accuracies = []
        model = BayesianNetwork(self.edges)
        best_acc = 0
        self.best_model = None

        kf = GroupKFold(n_splits=6)  # Đảm bảo args.GroupKFold là số nguyên

        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups = user_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            train_val_data = pd.concat([X_train_fold.reset_index(drop=True), 
                                    y_train_fold.reset_index(drop=True)], axis=1)
            test_val_data = pd.concat([X_val_fold.reset_index(drop=True), 
                                    y_val_fold.reset_index(drop=True)], axis=1)

            # Huấn luyện mô hình
            model.fit(train_val_data, estimator=ExpectationMaximization)

            # Dự đoán trên tập test
            infer = VariableElimination(model)
            y_pred = []

            for _, row in test_val_data.iterrows():
                evidence = row.drop(self.target, errors='ignore').to_dict()
                q = infer.map_query(variables=[self.target], evidence=evidence)

                y_pred.append(q[self.target])

            # Tính Accuracy
            acc = accuracy_score(test_val_data[self.target], y_pred)
            if acc > best_acc:
                best_acc = acc
                self.best_model = model
            accuracies.append(acc)
        
        path_EDA = '/kaggle/working/'
        EDA.draw_Bar(path_EDA, "BayesNetwork", accuracies, 'Accuracy Val')

    def predict(self, X_test, y_test):
        '''
        Predict with Bayesian Network
        '''
        infer = VariableElimination(self.best_model)
        y_pred = []
        test_set = pd.concat([X_test.reset_index(drop=True),y_test.reset_index(drop=True)], axis=1)

        for _, row in test_set.iterrows():
            evidence = row.drop(self.target, errors='ignore').to_dict()
            q = infer.map_query(variables=[self.target], evidence=evidence)

            y_pred.append(q[self.target])

        # Tính Accuracy
        acc = accuracy_score(test_set[self.target], y_pred)
        return acc
    
