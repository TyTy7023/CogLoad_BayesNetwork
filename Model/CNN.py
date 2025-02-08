import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

class CNNModel:
    def __init__(self, learning_rate=0.001):
        """
        Khởi tạo lớp CNNModel với các tham số.
        :param learning_rate: Learning rate cho optimizer.
        """
        self.learning_rate = learning_rate
        self.model = None

    def _build_model(self, input_shape):
        """Xây dựng kiến trúc mô hình CNN."""
        self.model = models.Sequential([
            layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
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
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler])

    def predict_proba(self, X_test):
        """
        Dự đoán kết quả với dữ liệu kiểm tra.
        :param X_test: Dữ liệu kiểm tra.
        :return: Kết quả dự đoán.
        """
        return self.model.predict(X_test)

    def predict(self, X_test):
        """
        Dự đoán kết quả với dữ liệu kiểm tra.
        :param X_test: Dữ liệu kiểm tra.
        :return: Kết quả dự đoán dưới dạng 0 1.
        """
        return  np.round(self.model.predict(X_test))

