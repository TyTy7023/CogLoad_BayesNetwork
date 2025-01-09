import pandas as pd
import numpy as np
from scipy.signal import resample
import warnings
import sys

'''Install library'''
sys.path.append('/kaggle/working/cogload/install_library')
from install_library import install_and_import
install_and_import("neurokit2")

import neurokit2 as nk

class eda():
    def __init__(self, gsr_df, num_hz):
        self.num_hz = num_hz
        self.gsr_df = gsr_df
        self.eda_features = self.extract_eda_features()
    
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
        self.eda_features.to_csv(path + 'eda_features.csv', index=False)
    
    def extract_eda_features(self):
        '''
        Extract EDA features from EDA data
        '''
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
            # Process it
            eda_signal = self.gsr_df.iloc[i].values
            resample_intervals = resample(eda_signal, int(len(eda_signal) * self.num_hz))
            signals, info = nk.eda_process(resample_intervals, sampling_rate=self.num_hz)

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
 