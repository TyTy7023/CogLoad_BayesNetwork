# Cog_BayesNetwork

## About My Data
- Gồm 3 thư mục (23_objects, allFeatures, last_30s_segments)
	+ 23_objects: dữ liệu thô của 23 đối tượng được ghi nhận trong quá trình thu thập dữ liệu. 
	+ allFeatures: gồm statFeatures.csv và features.csv
		*statFeatures.csv: 10 thống kê đặc trưng cơ bản (tham khảo từ bài báo gốc: [Colab_part-1](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)) bao gồm mean, standard deviation, skewness, kurtosis, diff, diff2, 25th quantile, 75th, quantile, qdev, max-min.
		*allFeatures.csv: bao gồm thống kê đặc trưng cơ bản và và các đặc trưng chuyên gia (tham khảo từ bài báo gốc [Colab_part-2](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing))
	+ last_30s_segments (_Dữ liệu làm việc chính_): gồm các tín hiệu sinh lý (gsr, hr, rr, temp) được trích xuất 30s cuối của mỗi tín hiệu và thông tin nhãn (labels) 

## Structure:
-  Exploratory_Data: 
    + Lớp EDA cung cấp các phương pháp trực quan hóa dữ liệu và so sánh hiệu suất của các mô hình. Các phương pháp này bao gồm vẽ biểu đồ ROC, biểu đồ cột, boxplot và biểu đồ đường.
- Model: 
    + EnsembleModel_7GB (E7GB) là một mô hình tập hợp (ensemble) gồm 7 mô hình LightGBM được cấu hình với các tham số khác nhau. Mục tiêu là cải thiện hiệu suất dự đoán thông qua việc kết hợp dự đoán của nhiều mô hình.
    + Ensemble SVM (ESVM) là mô hình kết hợp nhiều SVM với PCA để giảm chiều dữ liệu và AdaBoost để cải thiện độ chính xác. Mô hình sử dụng Optuna để tối ưu hóa các siêu tham số như C, gamma, kernel của SVM và tham số của AdaBoost, nhằm đạt được hiệu suất tối ưu cho bài toán phân loại
	+ TabNet là mô hình học sâu cho dữ liệu bảng, sử dụng attention để chọn lọc đặc trưng quan trọng. Mô hình này tối ưu hóa tự động qua quá trình huấn luyện và sử dụng Optuna để tối ưu siêu tham số, nhằm cải thiện độ chính xác trong phân loại.
	+ Multilayer Perception: 
		+ MLP Keras: Là một mô hình MLP sử dụng thư viện Keras với khả năng tối ưu hóa các siêu tham số bằng cách sử dụng RandomSearch từ keras_tuner. Mô hình này tìm kiếm các cấu hình tốt nhất cho số lượng lớp ẩn, số lượng nơ-ron trong mỗi lớp, thuật toán tối ưu, và các tham số khác. Optuna không được sử dụng trực tiếp trong phần này, thay vào đó keras_tuner sẽ tự động tối ưu hóa các tham số.
		+ MLP Sklearn: Sử dụng mô hình MLPClassifier từ sklearn với khả năng tìm kiếm các siêu tham số thông qua RandomizedSearchCV. Phương pháp này tối ưu hóa các tham số như số lượng lớp ẩn, hàm kích hoạt, thuật toán tối ưu và learning rate. RandomizedSearchCV sử dụng cross-validation với GroupKFold để tối ưu hóa các tham số và chọn mô hình tốt nhất.
	+ single_model: là file thực hiện training các mô hình đơn
	+ mul_model: là file thực hiện training các mô hình đa
	
- ProcessData:
    + Processing_Data: Cung cấp pipeline xử lý dữ liệu sinh lý với các bước: làm mượt dữ liệu bằng SMA, trích xuất đặc trưng thống kê (trung bình, độ lệch chuẩn, độ xiên, v.v.), loại bỏ đặc trưng không mong muốn, tách dữ liệu huấn luyện và kiểm tra theo nhóm người dùng, và chuẩn hóa dữ liệu (StandardScaler hoặc MinMaxScaler). Phương thức chính get_data() kết hợp toàn bộ quy trình để trả về tập dữ liệu và nhãn đã xử lý, sẵn sàng cho các mô hình học máy.
	+ Expert_HRV: Tính toán các đặc trưng HRV từ dữ liệu nhịp tim (RR intervals) sử dụng hrv-analysis và neurokit2
	+ Expert_eda: Tính toán các đặc trưng eda từ dữ liệu GSR (Galvanic Skin Response) sử dụng neurokit2 và PyTeAP libraries.
