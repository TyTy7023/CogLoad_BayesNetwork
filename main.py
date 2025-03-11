import os
import pandas as pd
import ast
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
parser.add_argument("--method", default = 'hill_climbing', type = str, help = "Method to draw DAG (hill_climbing or tabu_search)")

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
data = pd.read_csv(args.data_path + 'all_data.csv')
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

process = Processing(data, label_df, directory_name)
data = process.Discretization_of_data(bins = 2)

# create mean features
data = pd.read_csv(f'{directory_name}discrete_data.csv')
data['temp_features'] = data[['temp_mean', 'temp_std', 'temp_max-min', 'temp_skew', 'temp_kurtosis']].mean(axis=1)
data['hr_features'] = data[['hr_mean', 'hr_std', 'hr_max-min', 'hr_skew', 'hr_kurtosis']].mean(axis=1)
data['rr_features'] = data[['rr_mean', 'rr_std', 'rr_max-min', 'rr_skew', 'rr_kurtosis','HRV_Cd','HRV_AI','HRV_MedianNN','HRV_GI','HRV_CSI_Modified','HRV_CVNN','HRV_RMSSD','rr_diff2','HRV_IALS','HRV_PAS','HRV_HFn']].mean(axis=1) 
data['gsr_features'] = data[['gsr_mean', 'gsr_std', 'gsr_max-min', 'gsr_skew', 'gsr_kurtosis']].mean(axis=1)

selected_features.extend(['temp_features', 'hr_features', 'rr_features', 'gsr_features'])
data = data[selected_features]
cols = [col for col in data.columns if col != 'Labels'] + ['Labels']
data = data[cols]

#Processing data
process = Processing(data, label_df, directory_name)
X_train, y_train, X_test, y_test, user_train, user_test = process.get_Data(bins = 2)

# Draw DAG
bn = BN(data, method=args.method, path = directory_name)
bn.edges = [
    ('temp_mean', 'temp_features'), ('temp_std', 'temp_features'), ('temp_max-min', 'temp_features'), 
    ('temp_skew', 'temp_features'), ('temp_kurtosis', 'temp_features'),
    
    ('gsr_mean', 'gsr_features'), ('gsr_std', 'gsr_features'), ('gsr_max-min', 'gsr_features'), 
    ('gsr_skew', 'gsr_features'), ('gsr_kurtosis', 'gsr_features'),
    
    ('rr_mean', 'rr_features'), ('rr_std', 'rr_features'), ('rr_max-min', 'rr_features'), 
    ('rr_skew', 'rr_features'), ('rr_kurtosis', 'rr_features'), ('rr_diff2','rr_features'),
    
    ('HRV_Cd','rr_features'), ('HRV_AI','rr_features'), ('HRV_MedianNN','rr_features'), 
    ('HRV_GI','rr_features'), ('HRV_CSI_Modified','rr_features'), ('HRV_CVNN','rr_features'), 
    ('HRV_RMSSD','rr_features'), ('HRV_IALS','rr_features'), ('HRV_PAS','rr_features'), ('HRV_HFn','rr_features'),
    
    ('hr_mean', 'hr_features'), ('hr_std', 'hr_features'), ('hr_max-min', 'hr_features'), 
    ('hr_skew', 'hr_features'), ('hr_kurtosis', 'hr_features'), 
    
    ('hr_features', 'rr_features'),  
    ('rr_features', 'gsr_features'),  
    ('gsr_features', 'temp_features'), 

    ('rr_features', 'Labels'),('temp_features', 'Labels'),('gsr_features', 'Labels')
]
bn.fit(X_train, y_train, user_train, args.GroupKFold)
bn.predict(X_test, y_test)
# Save CPD result
bn.get_PDT()
