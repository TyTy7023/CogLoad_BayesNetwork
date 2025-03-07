import sys
import pandas as pd

sys.path.append('/kaggle/working/cogload/')
from install_library import install_and_import
install_and_import('pgmpy')

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
import networkx as nx

from sklearn.preprocessing import KBinsDiscretizer

class BayesianNetwork:
    def __init__(self, data, method='hill_climbing'):
        self.data = data
        self.method = method
        if method not in ['hill_climbing', 'tabu_search']:
            raise ValueError('Method not found')

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
    
    def fit(self, data, method='hill_climbing'):
        '''
        Fit Bayesian Network
        '''
        if method == 'hill_climbing':
            return self.Hill_climbing(data)
        elif method == 'tabu_search':
            return self.tabu_search(data)
        else:
            raise ValueError('Method not found')
        
    
