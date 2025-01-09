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
    def __init__(self, gsr_df):
        self.features = None
        self.resample_intervals = None
    
    def process_data(self):
        pass

    ''' EDA functions
    At the moment, we have 2 ways to extract EDA features:
    1. Using neurokit2 library
        We extract EDA features from EDA data to all features in neurokit2 library
        Then, we select the features which are not NaN in all 30-second EDA data
    2. Using PyTeAP library
    '''
    # Expert eda features by grs data
    def resample_eda(self):
        '''
        Resample GSR data to the same length as HR data
        '''
        # Resample GSR data to the same length as HR data
        gsr_resampled = resample(self.gsr_df.values, len(self.hr_df) * self.num_hz, axis=0)
        gsr_resampled = pd.DataFrame(gsr_resampled, columns=self.gsr_df.columns)
        return gsr_resampled
 