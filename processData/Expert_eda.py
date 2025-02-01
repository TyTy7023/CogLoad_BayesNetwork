import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import resample
import os
import warnings
import sys

'''Install library'''
sys.path.append('/kaggle/working/cogload/install_library')
from install_library import install_and_import
install_and_import("neurokit2")

sys.path.append('/kaggle/working/cogload/pyteap')
from gsr import acquire_gsr, get_gsr_features

import neurokit2 as nk

class eda():
    def __init__(self, gsr_df, num_hz):
        self.num_hz = num_hz
        self.gsr_df = gsr_df
        self.eda_features_nk = self.extract_eda_features()
        self.eda_features_pyteap = self.expert_eda_features()
    
    ''' EDA functions
    At the moment, we have 2 ways to extract EDA features:
    1. Using neurokit2 library
        We extract EDA features from EDA data to all features in neurokit2 library
        Then, we select the features which are not NaN in all 30-second EDA data
    2. Using PyTeAP library
    '''
    # Expert eda features by grs data
    def save_eda_features(self, path):
        '''
        Save EDA features to path
        '''
        self.eda_features_nk.to_csv(path + 'eda_features_nk.csv', index=False)
        self.eda_features_pyteap.to_csv(path + 'eda_features_pyteap.csv', index=False)
    
    def extract_eda_features(self):
        """
        Extract EDA features from EDA data
        """
        # EDA features to extract from neurokit2 library
        feature_keys = ['SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude', 'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime']

        #for each feature key we will caclulate min, max and mean values
        feature_names = []
        for f in feature_keys:
            feature_names.append('min_'+f)
            feature_names.append('max_'+f)
            feature_names.append('mean_'+f)

        #iterate through all 30-second segments
        features_arr = []
        for i in range(len(self.gsr_df)):
            my_eda = self.gsr_df.iloc[i].dropna()
            #Neurokit required 10Hz sampling frequency. Here we upsample the signal
            my_eda_resampled = resample(my_eda.values,len(my_eda.values)*10)
            # Process it
            signals, info = nk.eda_process(my_eda_resampled, sampling_rate=10)

            segment_features = []
            for k in feature_keys:
                #initial values are 0
                feature_min = 0
                feature_max = 0
                feature_mean = 0

                #drop Nan values
                values = info[k]
                values = values[~np.isnan(values)]
                if len(values)>0: #update feature-values if there is at least 1 detected value (e.g., at least one peak), else leave 0
                    feature_min = np.min(values)
                    feature_max = np.max(values)
                    feature_mean = np.mean(values)
                segment_features.extend([feature_min,feature_max,feature_mean])
            features_arr.append(segment_features)
        return pd.DataFrame(features_arr,columns = feature_names)
    
    ''' EDA functions with PyTeAP library'''
    def expert_eda_features(self):
        '''
        Extract EDA features using PyTeAP library
        '''
        # EDA features to extract from PyTeAP library
        eda = []
        for i in range(len(self.gsr_df)):
            eda.append(pd.Series(acquire_gsr(self.gsr_df.loc[i], 1)))
        feature_names = ['peaks_per_sec', 'mean_amp', 'mean_risetime', 'mean_gsr', 'std_gsr']
        features_arr = [get_gsr_features(eda[i], 1) for i in range(len(eda))]
        return pd.DataFrame(features_arr, columns=feature_names)