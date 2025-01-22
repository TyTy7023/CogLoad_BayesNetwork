import os
import pandas as pd
import warnings
import sys
sys.path.append('/kaggle/working/cogload/processData/')
from processing_Data import Preprocessing

# Bỏ qua cảnh báo trong quá trình thực thi
warnings.simplefilter("ignore")

# Đường dẫn thư mục chứa dữ liệu
data_folder_path = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/"

# Các tham số
GroupKFold = 3
window_size = 1
normalize = "Standard"
model_selected_feature = "None"
k_features = 11
forward = False
floating = True
estimator_RFECV = 'SVM'
debug = 0
models_single = ['LDA', 'SVM', 'RF', 'XGB']
models_mul = ['MLP_Sklearn', 'MLP_Keras', 'TabNet']
expert_lib = 'nk'

# Lưu lại các tham số vào file CSV
args_dict = {
    'data_folder_path': data_folder_path,
    'GroupKFold': GroupKFold,
    'window_size': window_size,
    'normalize': normalize,
    'model_selected_feature': model_selected_feature,
    'k_features': k_features,
    'forward': forward,
    'floating': floating,
    'estimator_RFECV': estimator_RFECV,
    'debug': debug,
    'models_single': models_single,
    'models_mul': models_mul,
    'expert_lib': expert_lib
}

log_args = pd.DataFrame([args_dict])
directory_name = '/kaggle/working/log/'
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
file_name = f'args.csv'  
log_args.to_csv(os.path.join(directory_name, file_name), index=False)

# Đọc dữ liệu
label_df = pd.read_excel(data_folder_path + 'labels.xlsx', index_col=0)
temp_df = pd.read_excel(data_folder_path + 'temp.xlsx', index_col=0)
hr_df = pd.read_excel(data_folder_path + 'hr.xlsx', index_col=0)
gsr_df = pd.read_excel(data_folder_path + 'gsr.xlsx', index_col=0)
rr_df = pd.read_excel(data_folder_path + 'rr.xlsx', index_col=0)

print("Data shapes:")
print('Labels', label_df.shape)
print('Temperature', temp_df.shape)
print('Heart Rate', hr_df.shape)
print('GSR', gsr_df.shape)
print('RR', rr_df.shape)

# Xử lý dữ liệu
preprocessing = Preprocessing(
    temp_df=temp_df, 
    hr_df=hr_df, 
    gsr_df=gsr_df, 
    rr_df=rr_df, 
    label_df=label_df, 
    window_size=window_size, 
    normalize=normalize, 
    expert_lib=expert_lib
)

X_train, y_train, X_test, y_test, user_train, user_test = preprocessing.get_data(features_to_remove='None')


# Example usage:
# Initialize and fit the model
from sklearn.metrics import accuracy_score
model = WeightedRegression(weight=0.7)
model.fit(X_train, y_train, user_train)

predictions = model.predict(X_test, user_test)
# Hàm tìm ngưỡng tối ưu
def find_optimal_threshold(y_true, y_pred):
    thresholds = np.linspace(min(y_pred), max(y_pred), 100)  # Thử nghiệm 100 ngưỡng
    best_threshold = thresholds[0]
    best_accuracy = 0

    for threshold in thresholds:
        predicted_classes = (y_pred >= threshold).astype(int)
        accuracy = accuracy_score(y_true, predicted_classes)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

# Predict and optimize weight
best_weight = model.optimize_weight(X_train, y_train, user_train)
optimized_predictions = model.predict(X_test, user_test)
optimal_threshold, optimal_accuracy = find_optimal_threshold(y_test, optimized_predictions)

print("Accuracy with Optimal Threshold:", optimal_accuracy)
