# visualization/plotter.py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ResultVisualizer:
    @staticmethod
    def plot_real_time_dashboard(measurements_history, attack_labels, pred_labels):
        """实时监控面板"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('节点电压监测', '攻击检测状态', '模型置信度'),
            vertical_spacing=0.1
        )
        
        # 提取数据
        timestamps = [m['timestamp'] for m in measurements_history]
        voltages = [m['bus_voltages'] for m in measurements_history]
        
        # 1. 电压曲线
        for bus_idx in range(min(5, len(voltages[0]))):  # 只显示前5个节点
            bus_voltage = [v[bus_idx] for v in voltages]
            fig.add_trace(
                go.Scatter(x=timestamps, y=bus_voltage, name=f'Bus {bus_idx+1}'),
                row=1, col=1
            )
        
        # 2. 攻击状态
        fig.add_trace(
            go.Scatter(x=timestamps, y=attack_labels, name='真实攻击', mode='lines+markers'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=pred_labels, name='检测结果', mode='lines+markers'),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=True)
        fig.show()
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix, save_path=None):
        """绘制混淆矩阵"""
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()