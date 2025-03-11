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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class BN:
    def __init__(self, data, method='hill_climbing', path='kaggle/working/'):
        self.path = path
        self.data = data
        self.target = 'Labels'
        self.method = method
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

    def fit(self, X_train, y_train, user_train, splits):
        accuracies = []
        y_true_list = []
        y_prob_list = []
        
        print("Edges of DAG:", self.edges)
        edges = list(self.edges)
        model = BayesianNetwork(edges)
        best_acc = 0
        self.best_model = None
    
        kf = GroupKFold(n_splits=splits)
    
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups=user_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            train_val_data = pd.concat([X_train_fold.reset_index(drop=True), 
                                    y_train_fold.reset_index(drop=True)], axis=1)
            test_val_data = pd.concat([X_val_fold.reset_index(drop=True), 
                                    y_val_fold.reset_index(drop=True)], axis=1)
            
            unique_nodes = set(node for edge in self.edges for node in edge)
            print('len unique nodes: ',len(unique_nodes))
            
            self.cols_drop = []
            columns = train_val_data.columns
            for col in columns:
                if col not in list(unique_nodes):
                    self.cols_drop.append(col)
            train_val_data = train_val_data.drop(columns = self.cols_drop)
            test_val_data = test_val_data.drop(columns = self.cols_drop)
            print('Shape train test val: ',train_val_data.shape,test_val_data.shape)
            
            # Huấn luyện mô hình
            model.fit(train_val_data, estimator=ExpectationMaximization)
    
            # Dự đoán trên tập test
            infer = VariableElimination(model)
            y_pred = []
            y_probs = []
    
            for _, row in test_val_data.iterrows():
                evidence = row.drop(self.target, errors='ignore').to_dict()
                q = infer.query(variables=[self.target], evidence=evidence)
                
                y_pred.append(q.values.argmax())  # Nhãn dự đoán
                y_probs.append(q.values[1])  # Xác suất của class 1
    
            acc = accuracy_score(test_val_data[self.target], y_pred)
            if acc > best_acc:
                best_acc = acc
                self.best_model = model
    
            accuracies.append(acc)
            y_true_list.append(test_val_data[self.target])
            y_prob_list.append(y_probs)

        EDA.draw_Bar(self.path, [f"fold {i+1}" for i in range(len(accuracies))] , accuracies, 'Accuracy Val')
        EDA.draw_ROC(self.path, y_true_list, y_prob_list, 'BN_Model')

    def get_PDT(self):
        '''
        Lấy bảng phân phối xác suất có điều kiện (CPD) của Bayesian Network
        '''
        if self.best_model is None:
            raise ValueError("Mô hình chưa được huấn luyện. Vui lòng chạy fit() trước.")
        
        pdt = {}
        for node in self.best_model.nodes():
            cpd = self.best_model.get_cpds(node)
            pdt[node] = cpd if cpd else "No CPD found"
        
        with open(f"{self.path}bayesian_network_cpd.txt", "w", encoding="utf-8") as f:
            for node, cpd in pdt.items():
                f.write(f"CPD của {node}:\n{cpd}\n\n")
                print(f"CPD của {node}:\n{cpd}\n")

    def predict(self, X_test, y_test):
        '''
        Predict with Bayesian Network
        '''
        infer = VariableElimination(self.best_model)
        y_pred = []
        test_set = pd.concat([X_test.reset_index(drop=True),y_test.reset_index(drop=True)], axis=1)
        test_set = test_set.drop(columns = self.cols_drop)

        y_probs = []
        for _, row in test_set.iterrows():
            evidence = row.drop(self.target, errors='ignore').to_dict()
            q = infer.query(variables=[self.target], evidence=evidence)
            
            y_pred.append(q.values.argmax())  # Nhãn dự đoán
            y_probs.append(q.values[1])  # Xác suất của class 1

        print(accuracy_score(test_set[self.target], y_pred))
        EDA.draw_ROC(self.path,test_set[self.target], [y_probs],['BN_model'])
        EDA.draw_Bar(self.path, 'BN_model', [accuracy_score(test_set[self.target], y_pred)], 'Accuracy Test')
                     
    
