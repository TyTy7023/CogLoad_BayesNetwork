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
        feature_keys = [
            'SCR_Onsets', 'SCR_Peaks', 'SCR_Height',
            'SCR_Amplitude', 'SCR_RiseTime', 
            'SCR_Recovery', 'SCR_RecoveryTime'
        ]
        
        # Generate feature names
        feature_names = [f"{stat}_{key}" for key in feature_keys for stat in ['min', 'max', 'mean']]
        
        # Process a single EDA signal and extract features 
        def process_segment(eda_signal):
            """
            Process a single EDA signal and extract features.
            """
            resample_intervals = resample(eda_signal, int(len(eda_signal) * self.num_hz))
            signals, info = nk.eda_process(resample_intervals, sampling_rate=self.num_hz)
            
            segment_features = []
            for key in feature_keys:
                values = np.array(info.get(key, []))  # Safely get the key or an empty array
                values = values[~np.isnan(values)]  # Remove NaN values
                if len(values) > 0:
                    segment_features.extend([np.min(values), np.max(values), np.mean(values)])
                else:
                    segment_features.extend([0, 0, 0])  # Default to 0 if no values
            return segment_features

        # Process all segments and construct DataFrame
        features_arr = [process_segment(self.gsr_df.iloc[i].values) for i in range(len(self.gsr_df))]
        return pd.DataFrame(features_arr, columns=feature_names)
    
    ''' EDA functions with PyTeAP library'''
    def expert_eda_features(self):
        '''
        Extract EDA features using PyTeAP library
        '''
        # EDA features to extract from PyTeAP library
        eda = []
        for i in range(len(self.gsr_df)):
            eda.append(pd.Series(acquire_gsr(self.gsr_df.loc[i], 1)))
        feature_names = ['peaks_per_sec', 'mean_amp', 'mean_risetime']
        features_arr = [get_gsr_features(eda[i], 1) for i in range(len(eda))]
        return pd.DataFrame(features_arr, columns=feature_names)