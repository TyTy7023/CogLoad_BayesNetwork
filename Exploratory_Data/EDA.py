import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


class EDA:
    @staticmethod
    # def draw_ROC_models_read_file(models, y_test,path):
    #     df = pd.read_csv(path)
    #     y_prob = []
    #     if path == '/kaggle/working/log/results_model.csv':
    #         df = pd.read_csv(path)
    #         # Xử lý để loại bỏ ký tự xuống dòng (\n)
    #         data_cleaned = df['Y Probs'].str.replace("\n", " ", regex=False)
    #         data_cleaned = data_cleaned.str.replace("[", "").str.replace("]", "")  # Loại bỏ dấu ngoặc vuông
    #         # Tách chuỗi và chuyển thành mảng số thực (float)
    #         for i in range(len(df['Y Probs'].values)):
    #             cleaned_string = data_cleaned.iloc[i].replace(',', '').replace('[', '').replace(']', '').strip()
    #             y = np.array([float(x) for x in cleaned_string.split()])
    #             y_prob.append(y)
    #     else:
    #         # Chuyển trực tiếp thành mảng nếu không cần xử lý
    #         parsed_data = np.array(df['Y Probs'])
    #         for item in parsed_data:
    #             # Loại bỏ nháy đơn, nháy kép và dấu ngoặc vuông
    #             item_cleaned = item.strip("[]").replace('"', '').replace("'", "").split(', ')
    #             # Chuyển thành danh sách số thực
    #             prob_values = [float(x) for x in item_cleaned]
    #             y_prob.append(prob_values)
    #         y_prob = np.array(y_prob)  # Chuyển thành mảng NumPy 2D
    #     EDA.draw_ROC(f'/kaggle/working/log/remove/', y_test, y_prob, models)
    def draw_ROC_models_read_file(models, y_test, path):
        df = pd.read_csv(path)
        y_prob = []

        if path == '/kaggle/working/log/results_model.csv':
            # Xử lý để loại bỏ ký tự xuống dòng và các ký tự không cần thiết
            data_cleaned = df['Y Probs'].str.replace("\n", " ", regex=False)
            data_cleaned = data_cleaned.str.replace("[", "", regex=False).str.replace("]", "", regex=False)
            
            # Chuyển đổi chuỗi thành mảng số thực
            y_prob = data_cleaned.apply(lambda x: np.array([float(val) for val in x.split()])).to_list()
        else:
            # Xử lý khi không cần làm sạch
            y_prob = df['Y Probs'].apply(
                lambda x: np.array([float(val) for val in x.strip("[]").replace('"', '').replace("'", "").split(', ')])
            ).to_list()
        
        y_prob = np.array(y_prob)  # Chuyển thành mảng NumPy 2D
        EDA.draw_ROC(f'/kaggle/working/log/remove/', y_test, y_prob, models)


    @staticmethod
    def _save_plot(path, filename):
        """Lưu biểu đồ vào thư mục được chỉ định."""
        if not os.path.exists(path):
            os.makedirs(path)
        plt.tight_layout()
        plt.savefig(os.path.join(path, filename))
        plt.show()

    @staticmethod
    def draw_ROC(path, y_test, y_preds, model):
        plt.figure(figsize=(8, 8))
        for i, y_pred in enumerate(y_preds):
            if isinstance(y_test, list) and len(y_test) == len(y_preds):
                fpr, tpr, _ = roc_curve(y_test[i], y_pred)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model} fold({i}) (AUC = {roc_auc:.2f})')
            else:
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{model[i]} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        EDA._save_plot(path, f"ROC-{model}")

    @staticmethod
    def draw_Bar(path, model, results, Type):
        df = pd.DataFrame({'Model': model, Type: results})
        plt.figure(figsize=(10, 6))
        barplot = sns.barplot(x='Model', y=Type, data=df, palette='pastel')
        plt.title('Algorithm Comparison')
        plt.ylabel(f'{Type} (Test)')

        for p in barplot.patches:
            barplot.annotate(f'{p.get_height():.2f}',
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=12)
        EDA._save_plot(path, Type)

    @staticmethod
    def draw_BoxPlot(path, model, results, Type):
        df = pd.DataFrame({'Model': model, Type: results})
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Model', y=Type, data=df, palette='pastel')
        plt.title('Algorithm Comparison')
        plt.ylabel(f'{Type} (Test)')
        EDA._save_plot(path, Type)

    @staticmethod
    def draw_LinePlot(path, model, results, Type):
        df = pd.DataFrame({'Feature': model, Type: results})
        plt.figure(figsize=(15, 6))
        sns.lineplot(x='Feature', y=Type, data=df, marker='o', palette='pastel', color='#FF6600')
        plt.title('Feature Importance')
        plt.ylabel(f'{Type} (Test)')
        plt.xticks(rotation=90)
        EDA._save_plot(path, Type)

    @staticmethod
    def draw_3D_Data(path, data):
        depth = data.shape[0]  # Số lượng lát cắt (632)
        rows = int(np.ceil(np.sqrt(depth)))  # Số hàng cho lưới subplots
        cols = rows  # Số cột cho lưới subplots

        # Tạo hình ảnh subplots
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        axes = axes.flatten()  # Chuyển mảng axes thành 1D để dễ xử lý

        for i in range(depth):
            ax = axes[i]
            im = ax.imshow(data[i], aspect='auto', cmap='viridis')  # Hiển thị lát cắt i
            ax.set_title(f"Slice {i}", fontsize=8)
            ax.axis('off')  # Tắt trục tọa độ

        # Xóa các ô subplot thừa (nếu có)
        for i in range(depth, len(axes)):
            fig.delaxes(axes[i])

        # Thêm thanh màu chung (colorbar)
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

        # Hiển thị hình ảnh
        plt.tight_layout()
        EDA._save_plot(path, 'DATA_3D')

