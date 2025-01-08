import pandas as pd
import numpy as np
from scipy.signal import resample

import sys
sys.path.append('/kaggle/working/cogload/ProcessData')
from Processing_Data import Preprocessing

class Expert_GSR(Preprocessing):
    def __init__(self, temp_df, hr_df, gsr_df, rr_df, label_df, window_size = 1, normalize = "Standard", data_type='', num_hz = 10):
        super().__init__(temp_df, hr_df, gsr_df, rr_df, label_df, window_size, normalize, data_type)
        self.num_hz = num_hz
        self.stat_feat_all = None
        self.stat_feat_after = pd.concat([temp_df, hr_df, gsr_df, rr_df],axis=1)

    def resample_GSR(self):
        '''
        Resample GSR data to the same length as HR data
        '''
        # Resample GSR data to the same length as HR data
        gsr_resampled = resample(self.gsr_df.values, len(self.hr_df) * self.num_hz, axis=0)
        gsr_resampled = pd.DataFrame(gsr_resampled, columns=self.gsr_df.columns)
        return gsr_resampled
    


    