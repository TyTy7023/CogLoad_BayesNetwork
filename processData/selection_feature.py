import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
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
    def selected_SBS(X_train, X_test, y_train, y_test, user_train, user_test, models, features_number):
        # create folder and file result
        def create_directory(path):
            if not os.path.exists(path):
                os.makedirs(path)
            return path
    
        def save_results_to_csv(filepath, data, mode='w', header=True):
            pd.DataFrame(data).to_csv(filepath, mode=mode, header=header, index=False)
    
        # Create necessary directories
        base_dir = create_directory('/kaggle/working/log/remove/result/')
    
        result_file = f'{base_dir}result.csv'
        save_results_to_csv(result_file, {
            'Model': [],
            'Best Column': [],
            'Shape': [],
            'Accuracy': [],
            'F1 Score': [],
            'Precision': [],
            'Recall': [],
            'Confusion Matrix': [],
            'Y Probs': []
        })

        # create variable
        raw_train = X_train.copy(deep=True)
        raw_test = X_test.copy(deep=True)

        for model in models:
            test_accuracies = []
            REMAIN = []
            ACC = []
            Y_PROBS = []
            F1_SCORE = []
            RECALL = []
            PRECISION = []
            MATRIX = []

            X_train_cp = raw_train.copy(deep=True)
            X_test_cp = raw_test.copy(deep=True)
            X_train = X_train_cp.copy(deep=True)
            X_test = X_test_cp.copy(deep=True)

            print(f"MODEL: {model} - SHAPE: {X_train.shape}")

            i = 0
            directory_name = f'/kaggle/working/log/remove/{model}/'
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)

            if model == 'MLP_Keras':
                X_train, X_test = Feature_Selection.selected_SFS(X_train = X_train,
                                                X_test = X_test, 
                                                y_train = y_train,
                                                model = SVC(kernel='linear'),
                                                k_features = features_number, 
                                                forward = False,
                                                floating = True
                                                )
                                                
                train_model(X_train, 
                            y_train, 
                            X_test, 
                            y_test, 
                            user_train,
                            user_test,
                            feature_remove='---', 
                            n_splits=3, 
                            path = directory_name, 
                            debug = 0,
                            models = [model],
                            index_name = i)
                
                df = pd.read_csv(directory_name + f'{i}_results_model.csv')
                max_accuracy = df['accuracy'].max()
                name_max_accuracy = df.loc[df['accuracy'].idxmax(), ['features_remove', 'y_probs', 'f1_score', 'precision', 'recall', 'confusion_matrix']]


                REMAIN.append(X_train.columns)
                ACC.append(max_accuracy) 
                Y_PROBS.append(name_max_accuracy['y_probs'])
                F1_SCORE.append(name_max_accuracy['f1_score'])
                PRECISION.append(name_max_accuracy['precision'])
                RECALL.append(name_max_accuracy['recall'])
                MATRIX.append(name_max_accuracy['confusion_matrix'])

                test_accuracies.append((X_train.columns, max_accuracy, name_max_accuracy['y_probs'], name_max_accuracy['f1_score'], name_max_accuracy['precision'], name_max_accuracy['recall'], name_max_accuracy['confusion_matrix'])) 
                
            else:
                if (model == 'ESVM' or model == 'WGLR') and X_train.shape[1] > 40:
                    X_train, X_test = Feature_Selection.selected_SFS(X_train = X_train,
                                                X_test = X_test, 
                                                y_train = y_train,
                                                model = SVC(kernel='linear'),
                                                k_features = 50, 
                                                forward = False,
                                                floating = True
                                                )
                    
                loop = X_train.shape[1] - 1
                features = X_train.columns.tolist()

                while(i<loop):
                    for feature in features:
                        X_train_cp = X_train.drop(columns=[f'{feature}'])
                        X_test_cp = X_test.drop(columns=[f'{feature}'])
                        
                        train_model(X_train_cp, 
                                        y_train, 
                                        X_test_cp, 
                                        y_test, 
                                        user_train,
                                        user_test,
                                        feature_remove=feature, 
                                        n_splits=3, 
                                        path = directory_name, 
                                        debug = 0,
                                        models = [model],
                                        index_name = i)
                            
                    df = pd.read_csv(directory_name + f'{i}_results_model.csv')
                    max_accuracy = df['accuracy'].max()
                    name_max_accuracy = df.loc[df['accuracy'].idxmax(), ['features_remove', 'y_probs', 'f1_score', 'precision', 'recall', 'confusion_matrix']]
                
                    X_train = X_train.drop(columns=[name_max_accuracy['features_remove']])
                    X_test = X_test.drop(columns=[name_max_accuracy['features_remove']])

                    REMAIN.append(X_train.columns)
                    ACC.append(max_accuracy) 
                    Y_PROBS.append(name_max_accuracy['y_probs'])
                    F1_SCORE.append(name_max_accuracy['f1_score'])
                    PRECISION.append(name_max_accuracy['precision'])
                    RECALL.append(name_max_accuracy['recall'])
                    MATRIX.append(name_max_accuracy['confusion_matrix'])

                    test_accuracies.append((X_train.columns, max_accuracy, name_max_accuracy['y_probs'], name_max_accuracy['f1_score'], name_max_accuracy['precision'], name_max_accuracy['recall'], name_max_accuracy['confusion_matrix'])) 
                    
                    features = X_train.columns.tolist() 
                    i += 1

            df = pd.DataFrame({'features': REMAIN, 'accuracy': ACC, 'y_probs': Y_PROBS, 'f1_score': F1_SCORE, 'precision': PRECISION, 'recall': RECALL, 'confusion_matrix': MATRIX})
            df.to_csv(f'/kaggle/working/log/remove/result/{model}.csv', index=False)
            
            feature_counts = [len(features) for features, _, _, _, _, _, _ in test_accuracies]
            accuracies = [accuracy for _, accuracy, _, _, _, _, _ in test_accuracies]
            
            plt.figure(figsize=(8, 5))
            plt.plot(feature_counts, accuracies, marker='o')
            plt.xlabel('Number of Features')
            plt.ylabel(f'Test Accuracy {model}')
            plt.title('Test Accuracy vs. Number of Features (Backward Selection)')
            plt.grid(True)
            plt.savefig(f'/kaggle/working/log/remove/result/{model}_acc.png')
            plt.show()
            
            best_column, max_accuracy, y_prob, f1_score, precision, recall, matrix = max(test_accuracies, key=lambda x: x[1])

            df_existing = pd.read_csv('/kaggle/working/log/remove/result/result.csv')
            if df_existing.empty: 
                df_to_append = pd.DataFrame({
                    'Model': model,
                    'Best Column': [best_column],
                    'Shape': len(best_column),
                    'Accuracy': max_accuracy,
                    'F1 Score': [f1_score],
                    'Precision': [precision],
                    'Recall': [recall],
                    'Confusion Matrix': [matrix],
                    'Y Probs': [y_prob]
                })
                df_to_append.to_csv('/kaggle/working/log/remove/result/result.csv', index=False)
            else:
                df_to_append = pd.DataFrame({
                    'Model': model,
                    'Best Column': [best_column],
                    'Shape': len(best_column),
                    'Accuracy': max_accuracy,
                    'F1 Score': [f1_score],
                    'Precision': [precision],
                    'Recall': [recall],
                    'Confusion Matrix': [matrix],
                    'Y Probs': [y_prob]
                }, columns=df_existing.columns)
            # Ghi thêm vào file CSV
                df_to_append.to_csv('/kaggle/working/log/remove/result/result.csv', mode='a', header=False, index=False)
