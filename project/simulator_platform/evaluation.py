# evaluation/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class Evaluator:
    def __init__(self):
        self.true_labels = []
        self.pred_labels = []
        self.detection_times = []
        
    def add_result(self, true_label, pred_label, detection_time):
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)
        self.detection_times.append(detection_time)
    
    def calculate_metrics(self):
        """计算所有评估指标"""
        if len(self.true_labels) == 0:
            return {}
        
        true_array = np.array(self.true_labels)
        pred_array = np.array(self.pred_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_array, pred_array, average='binary'
        )
        
        # 检测延迟（只计算攻击被检测到的情况）
        detection_delays = []
        for i in range(len(self.detection_times)):
            if self.true_labels[i] == 1 and self.pred_labels[i] == 1:
                detection_delays.append(self.detection_times[i])
        
        avg_delay = np.mean(detection_delays) if detection_delays else 0
        
        return {
            'accuracy': np.mean(true_array == pred_array),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': np.sum((true_array==0) & (pred_array==1)) / np.sum(true_array==0),
            'average_detection_delay': avg_delay,
            'total_samples': len(true_array),
            'attack_samples': int(np.sum(true_array))
        }
    
    def generate_report(self, output_path="./results/evaluation_report.txt"):
        """生成评估报告"""
        metrics = self.calculate_metrics()
        
        report = """
        ================================
        FDIA检测模型评估报告
        ================================
        总体统计:
        - 总样本数: {total_samples}
        - 攻击样本数: {attack_samples}
        - 正常样本数: {normal_samples}
        
        检测性能:
        - 准确率: {accuracy:.4f}
        - 精确率: {precision:.4f}
        - 召回率: {recall:.4f}
        - F1分数: {f1_score:.4f}
        - 误报率: {false_positive_rate:.4f}
        
        实时性能:
        - 平均检测延迟: {average_detection_delay:.4f} 秒
        ================================
        """.format(
            total_samples=metrics['total_samples'],
            attack_samples=metrics['attack_samples'],
            normal_samples=metrics['total_samples'] - metrics['attack_samples'],
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            false_positive_rate=metrics['false_positive_rate'],
            average_detection_delay=metrics['average_detection_delay']
        )
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(report)
        return metrics