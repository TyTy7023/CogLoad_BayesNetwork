import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.append('/kaggle/working/cogload/Exploratory_Data/')
from EDA import EDA 

sys.path.append('/kaggle/working/cogload/model/')

def train_model(X_train, y_train, X_test, y_test, user_train, user_test, path, feature_remove = ['None'], n_splits=3 , debug = 0, models = ['LDA', 'SVM', 'RF'], index_name = 1):
        # K-Fold Cross-Validation với 6 folds
    kf = GroupKFold(n_splits=n_splits)

    for model in models:
        log_results = []
        best_model = None
        best_score = 0
        y_vals = []
        y_pred_vals = []
        
        # Lặp qua từng fold
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train, groups = user_train)):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            y_vals.append(y_val_fold)

            
            id_user = np.array(user_train)
            # Kiểm tra nhóm trong fold
            train_groups = id_user[train_index]
            val_groups = id_user[val_index]

            # Train model
            if model == 'LDA':
                estimator = LDA(shrinkage = 0.5, solver = 'lsqr')     
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

            elif model == 'LR':
                estimator = LR(C= 1, penalty= 'l2', solver= 'liblinear', random_state=42)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

            elif model == 'SVM':
                estimator = SVC(kernel='rbf', C = 100, degree = 2, gamma = 0.001, probability=True, random_state=42)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

                estimator.fit(X_train_fold, y_train_fold)
            elif model == 'RF':
                estimator = RF(max_depth = 10, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 300)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

            elif model == 'XGB':
                estimator = XGBClassifier(colsample_bytree= 1.0, gamma= 0, learning_rate= 0.2, max_depth= 5, min_child_weight= 4, n_estimators= 100, subsample= 0.8, n_jobs=-1)
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]
            
            elif model == 'MLP_Sklearn':
                from Model.MLP_fix_param import MLP
                estimator = MLP.MLP_Sklearn()
                estimator.fit(X_train_fold, y_train_fold, train_groups)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]

            elif model == 'MLP_Keras':
                from Model.MLP_fix_param import MLP
                estimator = MLP.MLP_Keras()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, path)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            elif model == 'E7GB':
                from Model.E7GB import EnsembleModel_7GB
                estimator = EnsembleModel_7GB()
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            elif model == 'ESVM':
                from Model.ESVM_fix_param import ESVM
                # Tạo đối tượng ESVM
                estimator = ESVM()
                # Huấn luyện mô hình với dữ liệu
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]
            
            elif model == 'WGLR':
                from Model.WGLR import WeightedRegression
                estimator = WeightedRegression(weight=0.7)
                estimator.fit(X_train_fold, y_train_fold, train_groups)
                estimator.optimize_weight(X = X_train_fold,y = y_train_fold, group_ids = train_groups)
                y_pred_prob = estimator.predict_proba(X_val_fold, val_groups)

            elif model == 'TabNet':
                from Model.TabNet_fix_param import TabNet
                estimator = TabNet()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]

            else:
                raise ValueError(f"Model {model} is not supported")
            
            if model == 'WGLR':
                y_val_pred = estimator.predict(y_val_fold, y_pred_prob)
            else:
                y_val_pred = estimator.predict(X_val_fold)
            y_pred_vals.append(y_pred_prob)

            accuracy = accuracy_score(y_val_fold, y_val_pred)

            if accuracy > best_score:
                best_score = accuracy
                best_model = estimator

        if model == 'MLP_Keras' or model == 'E7GB':
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test).flatten()

        elif model == 'WGLR':
            y_prob = best_model.predict_proba(X_test, user_test)
            y_pred = best_model.predict(y_test, y_prob)

        else:
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:,1]

        # Đánh giá mô hình trên tập kiểm tra
        acc = accuracy_score(y_test, y_pred)
        precision =[precision_score(y_test, y_pred)]
        recall = [recall_score(y_test, y_pred)]
        f1 = [f1_score(y_test, y_pred)]
        matrix = [confusion_matrix(y_test, y_pred).tolist()]

        if not os.path.isfile(f'{path}{index_name}_results_model.csv'):
        # Tạo một DataFrame trống (nếu file cần chứa dữ liệu dạng bảng)
            df = pd.DataFrame({
                "model": model,
                "accuracy": f"{acc}",
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": matrix,
                "features_remove": [feature_remove],
                "y_probs": [y_prob.tolist()]
            })
            df.to_csv(f'{path}{index_name}_results_model.csv', index=False)

        df_existing = pd.read_csv(f'{path}{index_name}_results_model.csv')
        if df_existing.empty: 
            df_to_append = pd.DataFrame({
                "model": model,
                "accuracy": f"{acc}",
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": matrix,
                "features_remove": [feature_remove],
                "y_probs": [y_prob.tolist()]
            })
            df_to_append.to_csv(f'{path}{index_name}_results_model.csv', index=False)
        else:
            df_to_append = pd.DataFrame({
                "model": model,
                "accuracy": f"{acc}",
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": matrix,
                "features_remove": [feature_remove],
                "y_probs": [y_prob.tolist()]
            }, columns=df_existing.columns)
        # Ghi thêm vào file CSV
            df_to_append.to_csv(f'{path}{index_name}_results_model.csv', mode='a', header=False, index=False)
