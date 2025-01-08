import pandas as pd
import numpy as np
from scipy.signal import resample
import warnings
import sys

sys.path.append('/kaggle/working/cogload/install_library')
from install_library import install_and_import
install_and_import("hrv-analysis")
from hrvanalysis import get_time_domain_features, get_csi_cvi_features, get_frequency_domain_features, \
    get_geometrical_features, get_poincare_plot_features

sys.path.append('/kaggle/working/cogload/ProcessData')
from Processing_Data import Preprocessing

''' HRV functions '''
HRV_FUNCTIONS = [get_time_domain_features, get_csi_cvi_features,
                 lambda x: get_frequency_domain_features(x, sampling_frequency=1),
                 get_geometrical_features, get_poincare_plot_features]

class Expert_feature(Preprocessing):
    def __init__(self, temp_df, hr_df, gsr_df, rr_df, label_df, window_size = 1, normalize = "Standard", data_type='', num_hz = 10):
        super().__init__(temp_df, hr_df, gsr_df, rr_df, label_df, window_size, normalize, data_type)
        self.num_hz = num_hz
        self.stat_feat_all = None
        self.stat_feat_after = pd.concat([temp_df, hr_df, gsr_df, rr_df],axis=1)

    def resample_eda(self):
        '''
        Resample GSR data to the same length as HR data
        '''
        # Resample GSR data to the same length as HR data
        gsr_resampled = resample(self.gsr_df.values, len(self.hr_df) * self.num_hz, axis=0)
        gsr_resampled = pd.DataFrame(gsr_resampled, columns=self.gsr_df.columns)
        return gsr_resampled
    
    def get_all_hrv_features(self, ibis):
        """
        Wrapper to calculate hrv measures
        :param ibis: numpy array containing interbeat intervals (in ms)
        :return: pandas dataframe of hrv features
        """
        feats = [func(ibis) for func in HRV_FUNCTIONS]
        dataframes = []
        
        for feat in feats:
            if isinstance(feat, np.ndarray):  # Nếu là numpy.ndarray
                dataframes.append(pd.DataFrame(feat).T)  # Chuyển numpy array thành DataFrame
            elif isinstance(feat, dict):  # Nếu là dictionary
                dataframes.append(pd.DataFrame.from_dict(feat, orient='index').T)
        
        # Gộp các DataFrame lại
        result = pd.concat(dataframes, axis=1)
        return result

    def get_hrv_features(self):
        '''
        Get HRV features
        '''
        # get correct rr intervals in ms
        correct_rrs = self.rr_df.values * 1000
        hrv_list = []
        for i in range(len(correct_rrs)):
            # get hrv variables
            warnings.filterwarnings("ignore", category=UserWarning)
            hrv = self.get_all_hrv_features(correct_rrs[i])
            hrv = hrv.drop(['mean_hr', 'max_hr', 'min_hr', 'std_hr', 'tinn'], axis=1)
            hrv.columns = ['hrv__' + c for c in hrv.columns]
            warnings.filterwarnings("default", category=UserWarning)
            hrv_list.append(hrv)
        return pd.concat(hrv_list, axis=0, ignore_index=True).agg(list)


    