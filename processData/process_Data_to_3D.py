import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
sys.path.append('/kaggle/working/cogload/processData/')
from processing_Data import Preprocessing

class process3D_Data(Preprocessing):
    def __init__(self, temp_df, hr_df, gsr_df, rr_df, label_df, window_size = 1, normalize = "Standard", expert_lib='None'):
        super().__init__(temp_df, hr_df, gsr_df, rr_df, label_df, window_size, normalize, expert_lib)

    def normalize_data(self,X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Reshape dữ liệu từ 3D thành 2D
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_test = X_test.shape[0]
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        # Chuyển lại dữ liệu về 3D
        X_train_normalized = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
        X_test_normalized = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

        return X_train_normalized, X_test_normalized
    
    def splits_train_test(self):
        test_ids = ['3caqi','6frz4','bd47a','f1gjp','iz3x1']
        train_ids = ['1mpau', '2nxs5', '5gpsc', '7swyk', '8a1ep', 'b7mrd',
               'c24ur', 'dkhty', 'e4gay', 'ef5rq', 'f3j25', 'hpbxa',
               'ibvx8', 'iz2ps', 'rc1in', 'tn4vl', 'wjxci', 'yljm5']

        X_train = []
        y_train = []
        X_test = []
        y_test = []
        self.user_train = []
        self.user_test = []

        for user in self.label_df.user_id.unique():
            if user in train_ids:
                user_features = self.stat_feat_after[self.label_df.user_id == user]
                X_train.append(user_features)
                y = self.label_df.loc[self.label_df.user_id == user, 'level'].values

                # Convert labels (rest,0,1,2) to binary (rest vs task)
                y[y == 'rest'] = -1
                y = y.astype(int) + 1
                y[y > 0] = 1
                y_train.extend(y)

                temp = self.label_df.loc[self.label_df.user_id==user,'user_id'].values #labels
                self.user_train.extend(temp)
                
            elif user in test_ids:
                user_features = self.stat_feat_after[self.label_df.user_id == user]
                X_test.append(user_features)
                y = self.label_df.loc[self.label_df.user_id == user, 'level'].values

                # Convert labels (rest,0,1,2) to binary (rest vs task)
                y[y == 'rest'] = -1
                y = y.astype(int) + 1
                y[y > 0] = 1
                y_test.extend(y)

                temp = self.label_df.loc[self.label_df.user_id==user,'user_id'].values #labels
                self.user_test.extend(temp)

        # Concatenate and convert to DataFrame/NumPy array
        self.X_train = np.concatenate(X_train, axis=0)
        self.y_train = np.array(y_train)  
        self.X_test = np.concatenate(X_test, axis=0)
        self.y_test = np.array(y_test)  

    def get_data(self, features_to_remove):
        if self.stat_feat_all is None:
            if(self.window_size > 1):
                self.SMA()
            self.extract_features()
        
        self.expert_features()
        self.stat_feat_after = np.stack((self.temp_stat_features, self.hr_stat_features, self.gsr_stat_features, self.rr_stat_features), axis=-1)
        self.splits_train_test()
        self.X_train, self.X_test = self.normalize_data(self.X_train, self.X_test)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.user_train, self.user_test
