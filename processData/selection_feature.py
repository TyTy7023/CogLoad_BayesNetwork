import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.svm import SVC

import sys
sys.path.append('/kaggle/working/cogload/Train_Model/')
from model_fix_param import train_model 

class Feature_Selection:
    @staticmethod
    def selected_feature(selected_features, X_train, X_test):
        fs_train_orig = pd.DataFrame()
        fs_test_orig = pd.DataFrame()

        # Thay đổi cách nối cột và tên cột
        for i in selected_features:
            # Thêm cột vào fs_train và fs_test với tên cột tương ứng
            fs_train_orig[i] = X_train[i]
            fs_test_orig[i] = X_test[i]
        return fs_train_orig, fs_test_orig

    @staticmethod
    def selected_RFECV(X_train, X_test, y_train, user_train):
        gk = GroupKFold(n_splits=len(np.unique(user_train)))
        splits = gk.get_n_splits(X_train, y_train, user_train) #generate folds to evaluate the models using leave-one-subject-out
        fs_clf = RFECV(estimator=XGBClassifier(n_jobs=-1), #which estimator to use
                    step = 1, #how many features to be removed at each iteration
                    cv = splits,#use pre-defined splits for evaluation (LOSO)
                    scoring='accuracy',
                    min_features_to_select=1,
                    n_jobs=-1)
        fs_clf.fit(X_train, y_train)#perform feature selection. Depending on the size of the data and the estimator, this may last for a while
        selected_features = X_train.columns[fs_clf.ranking_==1]
        print(f"Selected feature : {selected_features}")
        return Feature_Selection.selected_feature(selected_features, X_train, X_test)

    @staticmethod
    def selected_SFS(X_train, X_test, y_train, model = SVC(kernel='linear'), k_features = 11, forward = False, floating = True):
        original_columns = list(X_train.columns)
        sfs = SFS(model, 
                k_features=k_features, 
                forward = forward, 
                floating = floating, 
                scoring = 'accuracy',
                cv = 3,
                n_jobs = -1)
        sfs = sfs.fit(X_train, y_train)
        selected_feature_indices = sfs.k_feature_idx_

        # Dùng chỉ số để lấy tên cột đã được chọn từ danh sách
        selected_features = [original_columns[i] for i in selected_feature_indices]
        print(f"Selected feature : {selected_features}")
        return Feature_Selection.selected_feature(selected_features, X_train, X_test)

    @staticmethod
    def selected_SBS(X_train, X_test, y_train, y_test, user_train, models, features_number):
        def create_directory(path):
            if not os.path.exists(path):
                os.makedirs(path)
            return path
    
        def save_results_to_csv(filepath, data, mode='w', header=True):
            pd.DataFrame(data).to_csv(filepath, mode=mode, header=header, index=False)
    
        # Create necessary directories
        base_dir = create_directory('/kaggle/working/log/remove/result')
    
        result_file = f'{base_dir}/result.csv'
        save_results_to_csv(result_file, {
            'Model': [], 'Best Column': [], 'Shape': [], 'Accuracy': [], 'Precision': [],
            'Recall': [], 'F1 Score': [], 'Confusion Matrix': [], 'Y Probs': []
        })
    
        raw_train, raw_test = X_train.copy(), X_test.copy()
    
        for model in models:
            test_accuracies = []
            remain, acc, y_probs = [], [], []
            X_train, X_test = raw_train.copy(), raw_test.copy()
            features = X_train.columns.tolist()
    
            print(f"MODEL: {model} - SHAPE: {X_train.shape}")
    
            model_dir = create_directory(f'/kaggle/working/log/remove/{model}/')
    
            if model == 'MLP_Keras':
                X_train, X_test = Feature_Selection.selected_SFS(
                    X_train=X_train, X_test=X_test, y_train=y_train,
                    model=SVC(kernel='linear'), k_features=features_number,
                    forward=False, floating=True
                )
    
                train_model(X_train, y_train, X_test, y_test, user_train,
                            feature_remove='---', n_splits=3, path=model_dir,
                            debug=0, models=[model], index_name=0)
    
                df = pd.read_csv(f'{model_dir}/0_results_model.csv')
                max_acc = df['accuracy'].max()
                best_feature, y_prob = df.loc[df['accuracy'].idxmax(), ['features_remove', 'y_probs']]
    
                remain.append(X_train.columns.tolist())
                acc.append(max_acc)
                y_probs.append(y_prob)
                test_accuracies.append((X_train.columns.tolist(), max_acc, y_prob))
            else:
                for i in range(len(features) - features_number):
                    feature_scores = []
                    for feature in features:
                        X_train_cp, X_test_cp = X_train.drop(columns=[feature]), X_test.drop(columns=[feature])
                        train_model(X_train_cp, y_train, X_test_cp, y_test, user_train,
                                    feature_remove=feature, n_splits=3, path=model_dir,
                                    debug=0, models=[model], index_name=i)
    
                        df = pd.read_csv(f'{model_dir}/{i}_results_model.csv')
                        feature_scores.append((
                            feature, 
                            df['accuracy'].max(),
                            df.loc[df['accuracy'].idxmax(), 'precision'],
                            df.loc[df['accuracy'].idxmax(), 'recall'],
                            df.loc[df['accuracy'].idxmax(), 'f1_score'],
                            df.loc[df['accuracy'].idxmax(), 'confusion_matrix'],
                            df.loc[df['accuracy'].idxmax(), 'y_probs']
                        ))
    
                    best_feature, max_acc, best_precision, best_recall, best_f1, best_confusion_matrix, y_prob = \
                        max(feature_scores, key=lambda x: x[1])
    
                    X_train, X_test = X_train.drop(columns=[best_feature]), X_test.drop(columns=[best_feature])
                    features.remove(best_feature)
    
                    remain.append(X_train.columns.tolist())
                    acc.append(max_acc)
                    y_probs.append(y_prob)
    
                    save_results_to_csv(f'{base_dir}/{model}.csv', {
                        'Feature Removed': [best_feature],
                        'Remaining Features': [X_train.columns.tolist()],
                        'Accuracy': [max_acc],
                        'Precision': [best_precision],
                        'Recall': [best_recall],
                        'F1 Score': [best_f1],
                        'Confusion Matrix': [best_confusion_matrix],
                        'Y Probs': [y_prob]
                    }, mode='a', header=False)
    
            if test_accuracies:
                feature_counts = [len(f) for f, _, _ in test_accuracies]
                accuracies = [a for _, a, _ in test_accuracies]
    
                plt.figure(figsize=(8, 5))
                plt.plot(feature_counts, accuracies, marker='o')
                plt.xlabel('Number of Features')
                plt.ylabel(f'Test Accuracy {model}')
                plt.title('Test Accuracy vs. Number of Features (Backward Selection)')
                plt.grid(True)
                plt.savefig(f'{base_dir}/{model}_acc.png')
                plt.close()
    
                best_features, max_accuracy, best_y_prob = max(test_accuracies, key=lambda x: x[1])
                save_results_to_csv(result_file, {
                    'Model': [model],
                    'Best Column': [best_features],
                    'Shape': [len(best_features)],
                    'Accuracy': [max_accuracy],
                    'Precision': [best_precision],
                    'Recall': [best_recall],
                    'F1 Score': [best_f1],
                    'Confusion Matrix': [best_confusion_matrix],
                    'Y Probs': [best_y_prob]
                }, mode='a', header=False)
