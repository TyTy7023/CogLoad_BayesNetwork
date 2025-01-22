#to access files and folders
import os
import ast
from datetime import datetime
#data analysis and manipulation library
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

import sys
sys.path.append('/kaggle/working/cogload/processData')
from processing_Data import Preprocessing

sys.path.append('/kaggle/working/cogload/Model')
from WGLR import WeightedRegression

#argument parser
parser = ArgumentParser()
parser.add_argument("--data_folder_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--window_size", default = 1, type = int, help = "Window size for feature extraction SMA")
parser.add_argument("--normalize", default = "Standard", type = str, help = "Normalization method, Standard or MinMax")
parser.add_argument("--model_selected_feature", default = "None", type = str, help = "None, RFECV, SFS, SBS")
parser.add_argument("--k_features", default = 11, type = int, help = "k of feature selected of SFS")
parser.add_argument("--forward", default = False, type = bool, help = "True to use backward, False to use forward")
parser.add_argument("--floating", default = True, type = bool, help = "True to use sfs with floating, False with no floating")
# parser.add_argument("--split", nargs='+', default=[1] , type=int, help="the split of data example 2 6 to split data into 2 and 6 to extract feature")
parser.add_argument("--estimator_RFECV", default='SVM', type=str, help="model for RFECV")
parser.add_argument("--debug", default = 0, type = int, help="debug mode 0: no debug, 1: debug")
# parser.add_argument("--models_single", nargs='+', default=[] , type=str, help="models to train, 'LDA', 'SVM', 'RF','XGB'")
# parser.add_argument("--models_mul", nargs='+', default=[] , type=str, help="models to train, 'MLP_Sklearn', 'MLP_Keras','TabNet'")
parser.add_argument("--models", nargs='+', default=[] , type=str, help="models to train")
parser.add_argument("--expert_lib",default='None' , type=str, help=" is the library used to extract expert features (None, 'nk', 'analysis_pyteap', 'HRV_nk', 'HRV_analysis', 'EDA_nk', 'pyteap', 'both')")

args = parser.parse_args()

args_dict = vars(args)
log_args = pd.DataFrame([args_dict])

directory_name = '/kaggle/working/log/'
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
file_name = f'args.csv'  
log_args.to_csv(os.path.join(directory_name, file_name), index=False)

#read the data
label_df = pd.read_excel(args.data_folder_path+'labels.xlsx',index_col=0)
temp_df= pd.read_excel(args.data_folder_path+'temp.xlsx',index_col=0)
hr_df= pd.read_excel(args.data_folder_path+'hr.xlsx',index_col=0)
gsr_df = pd.read_excel(args.data_folder_path+'gsr.xlsx',index_col=0)
rr_df= pd.read_excel(args.data_folder_path+'rr.xlsx',index_col=0)
print("Data shapes:")
print('Labels',label_df.shape)
print('Temperature',temp_df.shape)
print('Heart Rate',hr_df.shape)
print('GSR',gsr_df.shape)
print('RR',rr_df.shape)

# Xử lý dữ liệu
preprocessing = Preprocessing(temp_df = temp_df, 
                              hr_df = hr_df, 
                              gsr_df = gsr_df, 
                              rr_df = rr_df, 
                              label_df = label_df, 
                              window_size = args.window_size, 
                              normalize = args.normalize, 
                              expert_lib= args.expert_lib)
X_train, y_train, X_test, y_test, user_train, user_test = preprocessing.get_data(features_to_remove='None')

# Example usage:
# Initialize and fit the model
from sklearn.metrics import accuracy_score
model = WeightedRegression(weight=0.7)
model.fit(X_train, y_train, user_train)

predictions = model.predict(X_test, user_test)
# Hàm tìm ngưỡng tối ưu
def find_optimal_threshold(y_true, y_pred):
    thresholds = np.linspace(min(y_pred), max(y_pred), 100)  # Thử nghiệm 100 ngưỡng
    best_threshold = thresholds[0]
    best_accuracy = 0

    for threshold in thresholds:
        predicted_classes = (y_pred >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predicted_classes)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

# Predict and optimize weight
best_weight = model.optimize_weight(X_train, y_train, user_train)
optimized_predictions = model.predict(X_test, user_test)
optimal_threshold, optimal_accuracy = find_optimal_threshold(y_test, optimized_predictions)

print("Accuracy with Optimal Threshold:", optimal_accuracy)
