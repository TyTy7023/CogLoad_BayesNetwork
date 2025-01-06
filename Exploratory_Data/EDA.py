import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


class EDA:
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
