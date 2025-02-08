# Cog_BayesNetwork

## About My Data
- Gồm 3 thư mục (23_objects, allFeatures, last_30s_segments)
	+ 23_objects: dữ liệu thô của 23 đối tượng được ghi nhận trong quá trình thu thập dữ liệu. 
	+ allFeatures: gồm statFeatures.csv và features.csv
		*statFeatures.csv: 10 thống kê đặc trưng cơ bản (tham khảo từ bài báo gốc: [Colab_part-1](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)) bao gồm mean, standard deviation, skewness, kurtosis, diff, diff2, 25th quantile, 75th, quantile, qdev, max-min.
		*allFeatures.csv: bao gồm thống kê đặc trưng cơ bản và và các đặc trưng chuyên gia (tham khảo từ bài báo gốc [Colab_part-2](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing))
	+ last_30s_segments (_Dữ liệu làm việc chính_): gồm các tín hiệu sinh lý (gsr, hr, rr, temp) được trích xuất 30s cuối của mỗi tín hiệu và thông tin nhãn (labels) 

## Structure:
- Experiment: Thư mục này là nơi lưu trữ các kết quả thực nghiệm model

-  Exploratory_Data: 
    + Lớp EDA cung cấp các phương pháp trực quan hóa dữ liệu và so sánh hiệu suất của các mô hình. Các phương pháp này bao gồm vẽ biểu đồ ROC, biểu đồ cột, boxplot và biểu đồ đường.

- Model: Thư mục này chứa các tệp Python định nghĩa và triển khai các mô hình machine learning và deep learning.
    + CNN.py - Mạng nơ-ron tích chập (Convolutional Neural Network).
    + E7GB.py - Mô hình Ensemble Gradient Boosting (E7GB).
    + ESVM.py - Mô hình SVM (Support Vector Machine) cải tiến.
    + ESVM_fix_param.py - Mô hình SVM (Support Vector Machine) với tham số cố định.
    + MLP_fix_param.py - Mạng nơ-ron MLP (Multi-Layer Perceptron) với tham số cố định.
    + MLP.py - Mô hình MLP với các tham số có thể điều chỉnh.
    + RNN.py - Mạng nơ-ron hồi tiếp (Recurrent Neural Network).
    + TabNet_fix_param.py - Mô hình TabNet với tham số cố định.
    + TabNet.py - Mô hình TabNet có thể điều chỉnh.
    + WGLR.py - Mô hình hồi quy logistic tổng quát với trọng số (Weighted Generalized Logistic Regression).
	
- ProcessData: Thư mục này chứa các tệp xử lý dữ liệu trước khi đưa vào mô hình học máy.
    + Processing_Data: Cung cấp pipeline xử lý dữ liệu sinh lý với các bước: làm mượt dữ liệu bằng SMA, trích xuất đặc trưng thống kê (trung bình, độ lệch chuẩn, độ xiên, v.v.), loại bỏ đặc trưng không mong muốn, tách dữ liệu huấn luyện và kiểm tra theo nhóm người dùng, và chuẩn hóa dữ liệu (StandardScaler hoặc MinMaxScaler). Phương thức chính get_data() kết hợp toàn bộ quy trình để trả về tập dữ liệu và nhãn đã xử lý, sẵn sàng cho các mô hình học máy.
	+ Expert_HRV: Tính toán các đặc trưng HRV từ dữ liệu nhịp tim (RR intervals) sử dụng hrv-analysis và neurokit2
	+ Expert_eda: Tính toán các đặc trưng eda từ dữ liệu GSR (Galvanic Skin Response) sử dụng neurokit2 và PyTeAP libraries.
    + process_Data_to_3D.py - Chuyển đổi dữ liệu thành dạng 3D để phù hợp với mô hình CNN và RNN.
    + selection_feature.py - Chọn lọc đặc trưng để tối ưu hóa hiệu suất mô hình.

- Train_Model: Thư mục này chứa các tệp liên quan đến quá trình huấn luyện mô hình machine learning và deep learning.
    + model_fix_param.py - Huấn luyện mô hình với các tham số cố định, được sử dụng cho việc huấn luyện sử dụng SBS.
    + mul_model.py - Triển khai huấn luyện các mô hình đa và mô hình ensemble.
    + Neural_Network.py - Huấn luyện mô hình mạng nơ-ron nhân tạo (Neural Network).
    + single_model.py - Huấn luyện một mô hình đơn.
