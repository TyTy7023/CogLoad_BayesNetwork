import pandas as pd
import numpy as np
from scipy.signal import resample
import warnings
import sys

sys.path.append('/kaggle/working/cogload/install_library')
from install_library import install_and_import
install_and_import("neurokit2")
install_and_import("hrv-analysis")
import neurokit2 as nk
from hrvanalysis import get_time_domain_features, get_csi_cvi_features, get_frequency_domain_features, \
    get_geometrical_features, get_poincare_plot_features

sys.path.append('/kaggle/working/cogload/processData')
from processing_Data import Preprocessing

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
            hrv.columns = ['hrv_' + c for c in hrv.columns]
            warnings.filterwarnings("default", category=UserWarning)
            hrv_list.append(hrv)
        return pd.concat(hrv_list, axis=0, ignore_index=True).agg(list)

    


    def extract_HRV_features(self):
        feature_names = ['HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD', 'HRV_CVNN',
            'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN',
            'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF', 'HRV_VLF',
            'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
            'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI',
            'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS',
            'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
            'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
            'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn', 'HRV_SampEn',
            'HRV_MSE', 'HRV_CMSE', 'HRV_RCMSE', 'HRV_DFA', 'HRV_CorrDim']

        features_arr = []
        for i in range(len(self.rr_df)):

            try: #to avoid exceptions
                rr = self.rr_df.iloc[i].dropna() #30-second rr intervals

                #convert rr intervals to peaks array (input expected by neurokit)
                peaks_rr = np.zeros((len(rr)+1)*1000)
                peaks_rr[0]=1
                prev_peak = 0
                for r in rr:
                    peak_idx = prev_peak+int(r*1000)
                    prev_peak = peak_idx
                    peaks_rr[peak_idx]=1

                segment_features = nk.hrv(peaks_rr, sampling_rate=1000, show=False)
                features_arr.append(segment_features)
            except Exception as e: #when exception happens, fill-in with zeros
                values=np.zeros(len(feature_names))
                segment_features = pd.DataFrame([values],columns=feature_names)
                features_arr.append(segment_features)

        hrv_features = pd.concat(features_arr)
        hrv_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        good_features = (hrv_features.isnull().sum()==0)
        hrv_features = hrv_features[hrv_features.columns[good_features]]

        return hrv_features
