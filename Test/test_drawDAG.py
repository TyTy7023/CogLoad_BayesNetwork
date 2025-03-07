#to access files and folders
import os
#data analysis and manipulation library
import pandas as pd
from argparse import ArgumentParser

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

import sys
sys.path.append('/kaggle/working/cogload/BN_CognitiveLoad/')
from ProcessingData import Processing
from BN import BayesianNetwork



#argument parser
parser = ArgumentParser()
parser.add_argument("--data_labels_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--data_path", default = "/kaggle/input/cognitiveload/Feature_selection/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--method", default = 'hill_climbing', type = str, help = "Method to draw DAG (hill_climbing or tabu_search)")

args = parser.parse_args()

args_dict = vars(args)
log_args = pd.DataFrame([args_dict])

directory_name = '/kaggle/working/log/'
directory_result = '/kaggle/working/result/'
if not os.path.exists(directory_name):
    os.makedirs(directory_result)
    os.makedirs(directory_name)
file_name = f'args.csv'  
log_args.to_csv(os.path.join(directory_name, file_name), index=False)

#read the data
label_df = pd.read_excel(args.data_labels_path + 'labels.xlsx',index_col=0)
data = pd.read_csv(args.data_path + 'discrete_data.csv')
print("Data shapes:")
print('Labels',label_df.shape)
print('Data',data.shape)

#Processing data
process = Processing(data, label_df)
X_train, y_train, X_test, y_test, user_train, user_test = process.get_Data()

# Draw DAG
bn = BayesianNetwork(data, method=args.method)
edges = bn.fit(data, method=args.method)

print("Edges of DAG:", edges)

# train model
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

accuracies = []
model = BayesianNetwork(edges)
target = 'Labels'

kf = GroupKFold(n_splits=args.GroupKFold)  # Đảm bảo args.GroupKFold là số nguyên

for train_idx, test_idx in kf.split(X_train, y_train, groups=user_train):
    # Lấy dữ liệu theo index của fold hiện tại
    train_val_data = pd.concat([X_train.iloc[train_idx], y_train[train_idx]], axis=1)
    test_val_data = pd.concat([X_train.iloc[test_idx], y_train[test_idx]], axis=1)

    # Huấn luyện mô hình
    model.fit(train_val_data, estimator=ExpectationMaximization)

    # Dự đoán trên tập test
    infer = VariableElimination(model)
    y_pred = []

    for _, row in test_val_data.iterrows():
        evidence = row.drop(columns=[target]).to_dict()  # Xóa cột target để làm bằng chứng
        q = infer.map_query(variables=[target], evidence=evidence)
        y_pred.append(q[target])

    # Tính Accuracy
    acc = accuracy_score(test_val_data[target], y_pred)
    accuracies.append(acc)

print("Accuracies in val :", accuracies)
print("Mean accuracy in val:", np.mean(accuracies))




