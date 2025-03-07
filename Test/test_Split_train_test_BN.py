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


#argument parser
parser = ArgumentParser()
parser.add_argument("--data_labels_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--data_path", default = "/kaggle/input/cognitiveload/Feature_selection/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")

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
data = pd.read_csv(args.data_path + 'discrete_data.csv',index_col=0)
print("Data shapes:")
print('Labels',label_df.shape)
print('Data',data.shape)

#Processing data
process = Processing(data, label_df)
X_train, y_train, X_test, y_test = process.get_Data()
print("Data shapes:")
print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_test',X_test.shape)
print('y_test',y_test.shape)
