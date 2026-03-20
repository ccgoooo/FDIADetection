import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from detect_model import CNNTrainer, load_data  # 导入你的类和方法
import torch
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)

    # 1. 加载数据（与之前一致）
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("./processed_data/single_point_84")
    unique, counts = np.unique(y_test, return_counts=True)
    print("测试集类别分布：")
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 样本")

    
    # 如果数据需要重塑，保持与 main 中相同的逻辑
    # ...（这里可以复用 detect_model.py 中的 reshape 代码）

    # 2. 定义要比较的模型
    models_to_compare = [
        {"name": "SimpleCNN",         "type": "SimpleCNN"},          # 你需要让 CNNTrainer 支持新模型
        {"name": "ResidualCNN",       "type": "ResidualCNN"},
        # {"name": "DeepResidualCNN",   "type": "DeepResidualCNN"},
        {"name": "LSTM",              "type": "LSTM"},
        {"name": "ConvLSTM",          "type": "ConvLSTM"},
        {"name": "LightweightTransformer", "type": "LightweightTransformer"}
    ]

    results = []

    for model_info in models_to_compare:
        print(f"\n========== 训练模型：{model_info['name']} ==========")
        
        # 创建训练器
        trainer = CNNTrainer()
        
        # 计算类别权重（可选，但应保持一致）
        class_weights = trainer.compute_class_weights(y_train)
        
        # 准备数据加载器（批量大小统一为32）
        train_loader, val_loader, test_loader = trainer.prepare_data(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32
        )
        
        # 创建模型
        trainer.create_model(
            model_type=model_info["type"],
            feature_dim=X_train.shape[2],   # 根据数据形状
            window_size=X_train.shape[1],
            dropout_rate=0.3
        )
        
        # 记录参数量
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"参数量: {total_params:,}")
        
        # 训练并计时
        start_time = time.time()
        history = trainer.train(
            train_loader, val_loader,
            epochs=50,
            learning_rate=0.001,
            patience=10,
            class_weights=class_weights
        )
        train_time = time.time() - start_time
        print(f"训练耗时: {train_time:.2f} 秒")
        
        # 评估
        predictions, targets, probabilities = trainer.evaluate(test_loader)
        print("预测标签分布：")
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        for label, count in zip(unique_pred, counts_pred):
            print(f"  类别 {label}: {count} 样本")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, predictions)
        print("混淆矩阵：")
        print(cm)
        # 计算指标
        acc = accuracy_score(targets, predictions)
        prec = precision_score(targets, predictions)
        rec = recall_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        auc = roc_auc_score(targets, probabilities)
        
        results.append({
            "Model": model_info["name"],
            "Params": total_params,
            "Train Time (s)": round(train_time, 2),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "AUC": round(auc, 4)
        })
        
        # 可选：保存每个模型的训练历史图、混淆矩阵等
        # 可以复用 visualize_results 函数，但注意避免覆盖
        # visualize_results(history, predictions, targets, probabilities, model_info["name"])
    
    # 输出对比表格
    df_results = pd.DataFrame(results)
    print("\n\n========== 模型对比结果 ==========")
    print(df_results.to_string(index=False))
    
    # 保存到CSV
    df_results.to_csv("model_comparison.csv", index=False)
    
    # 可选：绘制柱状图
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt

    def add_value_labels(ax, spacing=5, fmt=".3f"):
        """
        为柱状图的每个柱子添加数值标签
        :param ax: matplotlib 坐标轴对象
        :param spacing: 标签与柱子顶部的距离（点）
        :param fmt: 数值格式，如 '.3f' 表示保留三位小数
        """
        for rect in ax.patches:
            height = rect.get_height()
            if not np.isnan(height):  # 确保数值有效
                ax.annotate(f'{height:{fmt}}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, spacing),  # 垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='black')

    # 第一组：Accuracy, Precision, Recall
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics1 = ["Accuracy", "Precision", "Recall","F1"]

    for i, metric in enumerate(metrics1):
        ax = axes.flat[i]
        bars = ax.bar(df_results["Model"], df_results[metric], color='skyblue', edgecolor='navy')
        ax.set_title(metric, fontsize=12)
        ax.set_xticklabels(df_results["Model"], rotation=45, ha='right')
        ax.set_ylim(0, 1.1)  # 预留空间给标签
        add_value_labels(ax, fmt=".3f")  # 保留三位小数

    plt.tight_layout()
    plt.savefig("model_comparison_metrics1.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 第二组：F1, AUC, Train Time (s)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    metrics2 = ["AUC", "Train Time (s)"]

    for i, metric in enumerate(metrics2):
        ax = axes[i]
        bars = ax.bar(df_results["Model"], df_results[metric], color='lightcoral', edgecolor='darkred')
        ax.set_title(metric, fontsize=12)
        ax.set_xticklabels(df_results["Model"], rotation=45, ha='right')

        # 根据数值范围调整标签格式和 y 轴上限
        if metric == "Train Time (s)":
            add_value_labels(ax, fmt=".1f")  # 时间保留一位小数
            ax.set_ylim(0, df_results[metric].max() * 1.15)
        else:
            add_value_labels(ax, fmt=".3f")
            ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("model_comparison_metrics2.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()