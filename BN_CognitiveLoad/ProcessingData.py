import sys
import pandas as pd
import numpy as np

sys.path.append('/kaggle/working/cogload/')
from install_library import install_and_import
install_and_import('pgmpy')

from sklearn.preprocessing import KBinsDiscretizer

class Processing:
    def __init__(self, data, labels):
        self.data = data.iloc[:, :-1]
        self.target = data['Labels']
        self.label_df = labels
        self.discrete_data = None

    def Discretization_of_data(self, bins = 2):
        num_cols = self.data.select_dtypes(include=['number']).columns  # Chỉ lấy cột số
        discretizer = KBinsDiscretizer(n_bins = bins, encode='ordinal', strategy='quantile')

        # Áp dụng rời rạc hóa
        self.discrete_data = pd.DataFrame(discretizer.fit_transform(self.data[num_cols]), columns=num_cols).astype(int)

        data = pd.concat([self.discrete_data, self.target], axis = 1)
        data.to_csv('/kaggle/working/discrete_data.csv', index=False)


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
                user_features = self.discrete_data[self.label_df.user_id == user]
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
                user_features = self.discrete_data[self.label_df.user_id == user]
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
        self.X_train = pd.concat(X_train)
        self.y_train = pd.DataFrame(y_train, columns=["Labels"])
        self.X_test = pd.concat(X_test)
        self.y_test = pd.DataFrame(y_test, columns=["Labels"])
    def get_Data(self, bins = 2):
        self.Discretization_of_data(bins)
        self.splits_train_test()
        return self.X_train, self.y_train, self.X_test, self.y_test, self.user_train, self.user_test 
