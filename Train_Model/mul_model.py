import os
import numpy as np
import pandas as pd
import itertools
import random

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import sys
sys.path.append('/kaggle/working/cogload/Exploratory_Data')
from EDA import EDA

sys.path.append('/kaggle/working/cogload/Model')
from MLP import MLP
from TabNet import TabNet
from E7GB import EnsembleModel_7GB
from ESVM import ESVM
from WGLR import WeightedRegression

def train_model(X_train, y_train, X_test, y_test, user_train, user_test, path, n_splits=3 , debug = 0, models = ['MLP_Sklearn', 'MLP_Keras','TabNet', 'E7GB', 'ESVM', 'WGLR']):
    np.random.seed(42)
    path = os.path.dirname(path)
    path_EDA = path + '/EDA/multi_model/'

    if debug == 1:
        models = models[:2]
    log_results = []
    test_accuracy_models = []
    accuracies_all = []
    f1_score_models = []
    y_pred_tests = []

    for model in models:
        print(f"\n\t\tMODEL: {model}")
        # K-Fold Cross-Validation với 6 folds
        kf = GroupKFold(n_splits=n_splits)

        best_model = None
        best_score = 0
        accuracy_all = []
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
            
            print(f'User of train_fold({fold}) : {np.unique(train_groups)}')
            print(f'User of val_fold({fold}) :{np.unique(val_groups)}')    

            # Train model
            if model == 'MLP_Sklearn':
                estimator = MLP.MLP_Sklearn()
                estimator.fit(X_train_fold, y_train_fold, train_groups)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]

            elif model == 'MLP_Keras':
                estimator = MLP.MLP_Keras()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, path)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            elif model == 'TabNet':
                estimator = TabNet()
                estimator.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:, 1]
            
            elif model == 'E7GB':
                estimator = EnsembleModel_7GB()
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)

            elif model == 'ESVM':
                estimator = ESVM()
                estimator.fit(X_train_fold, y_train_fold)
                y_pred_prob = estimator.predict_proba(X_val_fold)[:,1]

            elif model == 'WGLR':
                estimator = WeightedRegression(weight=0.7)
                estimator.fit(X_train_fold, y_train_fold, train_groups)
                estimator.optimize_weight(X = X_train_fold,y = y_train_fold, group_ids = train_groups)
                y_pred_prob = estimator.predict_proba(X_val_fold, val_groups)

            else:
                raise ValueError(f"Model {model} is not supported")
            
            y_pred_vals.append(y_pred_prob)
            if model == 'WGLR':
                y_val_pred = estimator.predict(y_val_fold, y_pred_prob)
            else:
                y_val_pred = estimator.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            accuracy_all.append(accuracy)

            if accuracy > best_score:
                best_score = accuracy
                best_model = estimator

        # ROC tâp validation K-Fold
        EDA.draw_ROC(path_EDA + "/models/", y_vals, y_pred_vals, model)

        # Dự đoán trên tập kiểm tra
        print(f"Best parameters found: {best_model.best_params}\n")
        if model == 'MLP_Keras' or model == 'E7GB':
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            y_pred_tests.append(y_pred_proba)
            
        if model == 'WGLR':
            y_pred_proba = best_model.predict_proba(X_test, user_test)
            y_pred = best_model.predict(y_test, y_pred_proba)
            y_pred_tests.append(y_pred_proba)

        else: 
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            y_pred_tests.append(y_pred_proba)

        # Đánh giá mô hình trên tập kiểm tra
        acc = accuracy_score(y_test, y_pred)
        precision =[precision_score(y_test, y_pred)]
        recall = [recall_score(y_test, y_pred)]
        matrix = [confusion_matrix(y_test, y_pred).tolist()]
        f1Score = f1_score(y_test, y_pred)

        test_accuracy_models.append(acc)
        accuracies_all.extend(accuracy_all)
        f1_score_models.append(f1Score.mean())

        print(f"ACCURACY: {acc}")

        accuracy_all = np.array(accuracy_all)
        print(f"Accuracy all fold: {accuracy_all}\nMean: {accuracy_all.mean()} ---- Std: {accuracy_all.std()}")

        log_results.append({
            "Model": model,
            "Accuracy": f"{acc} +- {accuracy_all.std()}",
            "best_model": best_model.best_params,
            "F1 Score": f1Score,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": matrix,
            'Y Probs': [y_pred_proba]
        })
        print("\n===================================================================================================================================\n")
    log_results = pd.DataFrame(log_results)
    file_name = f'results_multi_model.csv'  # Tên file tự động
    log_results.to_csv(os.path.join(path, file_name), index=False)

    EDA.draw_Bar(path_EDA, models, test_accuracy_models, 'Accuracy Test')
    EDA.draw_BoxPlot(path_EDA, list(itertools.chain.from_iterable([[i]*3 for i in models])), accuracies_all, 'Accuracy train')
    EDA.draw_Bar(path_EDA, models, f1_score_models, 'F1 Score')
    EDA.draw_ROC(path_EDA, y_test, y_pred_tests, models)

