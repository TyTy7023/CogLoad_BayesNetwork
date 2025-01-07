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
    + Raw_Model: xây dựng và đánh giá nhiều mô hình học máy (Logistic Regression, Random Forest, XGBoost, v.v.) sử dụng Group K-Fold Cross-Validation để tránh rò rỉ dữ liệu nhóm, hỗ trợ huấn luyện, tối ưu siêu tham số (GridSearchCV), đánh giá hiệu suất (Accuracy, F1, Log Loss) và trực quan hóa (ROC curves, bar plot, box plot)
- ProcessData:
    + Processing_Data: Cung cấp pipeline xử lý dữ liệu sinh lý với các bước: làm mượt dữ liệu bằng SMA, trích xuất đặc trưng thống kê (trung bình, độ lệch chuẩn, độ xiên, v.v.), loại bỏ đặc trưng không mong muốn, tách dữ liệu huấn luyện và kiểm tra theo nhóm người dùng, và chuẩn hóa dữ liệu (StandardScaler hoặc MinMaxScaler). Phương thức chính get_data() kết hợp toàn bộ quy trình để trả về tập dữ liệu và nhãn đã xử lý, sẵn sàng cho các mô hình học máy.
