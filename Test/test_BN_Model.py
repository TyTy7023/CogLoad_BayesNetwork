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
from BN import BN

sys.path.append('/kaggle/working/cogload/Exploratory_Data/')
from EDA import EDA

#argument parser
parser = ArgumentParser()
parser.add_argument("--data_labels_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--data_path", default = "/kaggle/input/cognitiveload/Feature_selection/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--method", default = 'None', type = str, help = "Method to draw DAG (hill_climbing or tabu_search)")
parser.add_argument("--edges", nargs='+', default=[] , type=tuple, help="models to train")


args = parser.parse_args()

args_dict = vars(args)
log_args = pd.DataFrame([args_dict])

directory_name = '/kaggle/working/log/'
if not os.path.exists(directory_name):
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
selected_features = ['temp_mean','temp_std','temp_max-min','temp_skew','temp_kurtosis',
                     'gsr_mean','gsr_std','gsr_max-min', 'gsr_skew', 'gsr_kurtosis',
                     'rr_mean','rr_std','rr_max-min','rr_skew','rr_kurtosis',
                     'hr_mean','hr_std','hr_max-min','hr_skew','hr_kurtosis',
                     'HRV_Cd','HRV_AI','HRV_MedianNN','HRV_GI','HRV_CSI_Modified', 
                     'HRV_CVNN', 'rr_diff2','HRV_RMSSD','HRV_IALS','HRV_PAS','HRV_HFn',
                     'Labels']  # Thay thế bằng các đặc trưng bạn muốn giữ lại
data = data[selected_features]

process = Processing(data, label_df)
data = process.Discretization_of_data(bins = 2)

# create new mean features
# Xác định nhóm đặc trưng
feature_groups = {
    "temp_features": ['temp_mean', 'temp_std', 'temp_max-min', 'temp_skew', 'temp_kurtosis'],
    "hr_features": ['hr_mean', 'hr_std', 'hr_max-min', 'hr_skew', 'hr_kurtosis'],
    "rr_features": ['rr_mean', 'rr_std', 'rr_max-min', 'rr_skew', 'rr_kurtosis', 
                    'HRV_Cd', 'HRV_AI',  'HRV_MedianNN', 'HRV_GI', 'HRV_CSI_Modified', 
                    'HRV_CVNN', 'HRV_RMSSD', 'rr_diff2', 
                    'HRV_IALS', 'HRV_PAS', 'HRV_HFn'],
    "gsr_features": ['gsr_mean', 'gsr_std', 'gsr_max-min', 'gsr_skew', 'gsr_kurtosis']
}

for feature_name, columns in feature_groups.items():
    data[feature_name] = data[columns].mean(axis=1, skipna=True)

#Processing data
process = Processing(data, label_df)
X_train, y_train, X_test, y_test, user_train, user_test = process.get_Data(bins = 2)

# Draw DAG
bn = BN(data, method=args.method, edges = args.edges)
bn.fit(X_train, y_train, user_train, args.GroupKFold)
accuracy = bn.predict(X_test, y_test)

# Save CPD result
bn.get_PDT()

# Draw DAG
EDA.draw_DAG(directory_name, bn.edges)