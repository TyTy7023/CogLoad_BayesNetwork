from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class RNNModel:
    def __init__(self, rnn_type='LSTM', units=64, learning_rate=0.001):
        """
        Khởi tạo lớp RNNModel với các tham số.
        :param rnn_type: Loại RNN ('SimpleRNN', 'LSTM', 'GRU').
        :param units: Số lượng đơn vị trong lớp RNN.
        :param learning_rate: Learning rate cho optimizer.
        """
        self.rnn_type = rnn_type
        self.units = units
        self.learning_rate = learning_rate
        self.model = None

    def _build_model(self, input_shape):
        """Xây dựng kiến trúc mô hình RNN."""
        rnn_layer = {
            'SimpleRNN': layers.SimpleRNN,
            'LSTM': layers.LSTM,
            'GRU': layers.GRU
        }.get(self.rnn_type, layers.LSTM)  # Mặc định là LSTM nếu không chỉ định

        self.model = models.Sequential([
            rnn_layer(self.units, activation='tanh', input_shape=input_shape, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Lớp đầu ra cho dự đoán nhị phân
        ])

    def compile_model(self):
        """Biên dịch mô hình."""
        lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        return lr_scheduler

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Huấn luyện mô hình.
        :param X_train: Dữ liệu huấn luyện.
        :param y_train: Nhãn huấn luyện.
        :param epochs: Số lượng epochs.
        :param batch_size: Kích thước batch.
        """
        # Lấy input_shape tự động từ X_train
        input_shape = X_train.shape[1:]  # Bỏ kích thước batch
        if self.model is None:
            self._build_model(input_shape)
        
        lr_scheduler = self.compile_model()
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler])
        return history

    def predict(self, X_test):
        """
        Dự đoán kết quả với dữ liệu kiểm tra.
        :param X_test: Dữ liệu kiểm tra.
        :return: Kết quả dự đoán.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình trên tập kiểm tra.
        :param X_test: Dữ liệu kiểm tra.
        :param y_test: Nhãn kiểm tra.
        :return: Độ chính xác và báo cáo phân loại.
        """
        pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, np.round(pred))
        report = classification_report(y_test, np.round(pred))
        return accuracy, report
