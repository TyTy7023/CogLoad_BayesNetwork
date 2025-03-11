# Cog_BayesNetwork

## About My Data
- Gồm 3 thư mục (23_objects, allFeatures, last_30s_segments)
	+ 23_objects: dữ liệu thô của 23 đối tượng được ghi nhận trong quá trình thu thập dữ liệu. 
	+ allFeatures: gồm statFeatures.csv và features.csv
		*statFeatures.csv: 10 thống kê đặc trưng cơ bản (tham khảo từ bài báo gốc: [Colab_part-1](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)) bao gồm mean, standard deviation, skewness, kurtosis, diff, diff2, 25th quantile, 75th, quantile, qdev, max-min.
		*allFeatures.csv: bao gồm thống kê đặc trưng cơ bản và và các đặc trưng chuyên gia (tham khảo từ bài báo gốc [Colab_part-2](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing))
	+ last_30s_segments (_Dữ liệu làm việc chính_): gồm các tín hiệu sinh lý (gsr, hr, rr, temp) được trích xuất 30s cuối của mỗi tín hiệu và thông tin nhãn (labels) 
- Sau khi qua xử lý chúng tôi đã điều chỉnh dữ liệu để phù hợp với mô hình Mạng Bayesian, tất cả đều được đẩy lên [DATASET](https://www.kaggle.com/datasets/quanminhminhquan/cognitiveload)
## Purpose:
- Xây dựng mô hình Mạng Bayesian dựa trên các đặc trưng được chọn nhằm nâng cao hiệu suất phân loại tín hiệu nhận thức. Nghiên cứu tập trung vào việc khai thác mối quan hệ giữa các tín hiệu sinh lý (RR, GSR, HR, Temp) và trạng thái tải nhận thức (Rest/Load) thông qua mô hình Bayesian
  <img src="https://learningpartnership.s3.amazonaws.com/uploads/asset_image/2_299.jpg" alt="CognitiveLoad" width="250" />
  ![CognitiveLoad](https://learningpartnership.s3.amazonaws.com/uploads/asset_image/2_299.jpg)
- Hướng thực hiện nghiên cứu 
  	+ Phân tích và đánh giá các phương pháp hiện có: Tiến hành phân tích và đánh giá các mô hình truyền thống như SVM, Random Forest, Gradient Boosting, CNN, RNN, v.v., nhằm xác định các phương pháp tối ưu trong phân loại tín hiệu nhận thức.
	+ Phát triển phương pháp mới: Đề xuất một mô hình dựa trên mạng Bayesian, trong đó các tín hiệu sinh lý (RR, GSR, HR, Temp) đóng vai trò là các nút cha, các đặc trưng trích xuất từ dữ liệu là nút con, và nhãn phân loại (Rest/Load) là đầu ra.
	+ Thử nghiệm và đánh giá: Đánh giá hiệu suất của mô hình mạng Bayesian nhằm đánh giá khả năng ứng dụng trong phân loại tín hiệu nhận thức.
