import warnings
warnings.filterwarnings('ignore', message='numba cannot be imported')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import os
from model import ResidualCNN,DeepResidualCNN,FastConvLSTM,LightweightTransformer
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑、黑体
plt.rcParams['axes.unicode_minus'] = False




# ================ 训练系统 ================
class CNNTrainer:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        print(f"使用设备: {self.device}")
    
    def create_model(self, model_type="ResidualCNN", feature_dim=84, window_size=10, dropout_rate=0.3):
        """创建模型"""
        if model_type == "DeepResidualCNN":
            self.model = DeepResidualCNN(
                window_size=window_size,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate
            ).to(self.device)
        if model_type == "ResidualCNN":
            self.model = ResidualCNN(
                window_size=window_size,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate
            ).to(self.device)
        if model_type == "ConvLSTM":
            self.model = FastConvLSTM(
                window_size=window_size,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate
            ).to(self.device)
        if model_type == "LightweightTransformer":
            self.model = LightweightTransformer(
                window_size=window_size,
                feature_dim=feature_dim,
                dropout_rate=dropout_rate
            ).to(self.device)
        
        print(f"模型创建成功: {model_type}")
        print(f"总参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def prepare_data(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """准备数据加载器"""
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"数据加载器准备完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def compute_class_weights(self, y_train):
        """计算类别权重以处理不平衡数据"""
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        
        # 计算类别权重（样本少的类别权重高）
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"类别分布: {dict(enumerate(class_counts))}")
        print(f"类别权重: {class_weights.cpu().numpy()}")
        
        return class_weights
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001, patience=10, class_weights=None):
        """训练模型"""
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_accuracy = 0
        best_model_state = None
        patience_counter = 0
        
        train_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        print(f"\n开始训练，共{epochs}个epochs")
        print("-" * 60)
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  新最佳验证准确率: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # 记录历史
            train_history['loss'].append(train_loss)
            train_history['accuracy'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_accuracy'].append(val_acc)
            
            # 打印进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                      f"LR: {current_lr:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停触发，在epoch {epoch+1}停止")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n训练完成，最佳验证准确率: {best_val_accuracy:.2f}%")
        
        return train_history
    
    def evaluate(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])  # 攻击类别的概率
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def save_model(self, path="models/best_residual_cnn.pth"):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model.__class__.__name__
        }, path)
        print(f"模型保存到 {path}")


# ================ 数据加载函数 ================
def load_data(data_dir="processed_data"):
    """加载已处理的数据"""
    print(f"从目录 {data_dir} 加载数据...")
    
    # 加载数据
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # 打印数据信息
    print(f"\n数据形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  y_test: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ================ 可视化函数 ================
def visualize_results(history, predictions, targets, probabilities, model_name="ResidualCNN"):
    """可视化训练结果"""
    
    # 1. 训练历史
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history['loss'], label='训练损失', linewidth=2)
    axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title(f'{model_name} - 训练和验证损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['accuracy'], label='训练准确率', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].set_title(f'{model_name} - 训练和验证准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/training_history_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 混淆矩阵和ROC曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 混淆矩阵
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{model_name} - 混淆矩阵')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('假正率')
    axes[1].set_ylabel('真正率')
    axes[1].set_title(f'{model_name} - ROC曲线')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/evaluation_results_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. 预测概率分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 概率直方图
    normal_probs = probabilities[targets == 0]
    attack_probs = probabilities[targets == 1]
    
    axes[0].hist(normal_probs, bins=30, alpha=0.7, label='正常样本', color='green')
    axes[0].hist(attack_probs, bins=30, alpha=0.7, label='攻击样本', color='red')
    axes[0].set_xlabel('攻击概率')
    axes[0].set_ylabel('样本数')
    axes[0].set_title(f'{model_name} - 预测概率分布')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 性能指标表格
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    metrics_data = [
        ["准确率", f"{accuracy:.4f}"],
        ["精确率", f"{precision:.4f}"],
        ["召回率", f"{recall:.4f}"],
        ["F1分数", f"{f1:.4f}"],
        ["AUC", f"{roc_auc:.4f}"]
    ]
    
    table = axes[1].table(cellText=metrics_data, 
                         colLabels=["指标", "数值"],
                         loc='center',
                         cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    axes[1].axis('off')
    axes[1].set_title(f'{model_name} - 性能指标')
    
    plt.tight_layout()
    plt.savefig(f'results/performance_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. 打印分类报告
    print(f"\n{model_name} - 详细分类报告:")
    print(classification_report(targets, predictions, target_names=['正常', '攻击']))


# ================ 主函数 ================
def main():
    """主函数 - 完整的残差CNN训练流程"""
    print("=" * 60)
    print("电力系统攻击检测 - 残差CNN模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n步骤1: 加载数据...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("./processed_data")
    
    # 检查数据是否需要重塑（如果是2D数据）
    if len(X_train.shape) == 2:
        print("\n检测到2D数据，自动重塑为3D格式...")
        # 假设窗口大小为10
        window_size = 10
        feature_dim = X_train.shape[1]
        
        # 计算可用的窗口数
        n_windows_train = X_train.shape[0] // window_size
        n_windows_val = X_val.shape[0] // window_size
        n_windows_test = X_test.shape[0] // window_size
        
        # 重塑数据
        X_train = X_train[:n_windows_train * window_size].reshape(n_windows_train, window_size, feature_dim)
        X_val = X_val[:n_windows_val * window_size].reshape(n_windows_val, window_size, feature_dim)
        X_test = X_test[:n_windows_test * window_size].reshape(n_windows_test, window_size, feature_dim)
        
        # 重塑标签（每个窗口取一个标签）
        y_train = y_train[:n_windows_train * window_size:window_size]
        y_val = y_val[:n_windows_val * window_size:window_size]
        y_test = y_test[:n_windows_test * window_size:window_size]
        
        print(f"重塑后形状:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 2. 创建训练器
    print("\n步骤2: 创建训练器...")
    trainer = CNNTrainer()
    
    # 3. 确定特征维度和窗口大小
    if len(X_train.shape) == 3:
        window_size = X_train.shape[1]
        feature_dim = X_train.shape[2]
    else:
        window_size = 1
        feature_dim = X_train.shape[1]
    
    print(f"\n步骤3: 确定模型参数")
    print(f"  特征维度: {feature_dim}")
    print(f"  窗口大小: {window_size}")
    
    # 4. 创建模型（可以选择ResidualCNN或DeepResidualCNN）
    print("\n步骤4: 创建模型...")
    model_type = "LightweightTransformer"  # 可以改为"DeepResidualCNN"\"convLSTM"\"LightweightTransformer"
    trainer.create_model(
        model_type=model_type,
        feature_dim=feature_dim,
        window_size=window_size,
        dropout_rate=0.3
    )
    
    # 5. 计算类别权重（处理不平衡数据）
    print("\n步骤5: 计算类别权重...")
    class_weights = trainer.compute_class_weights(y_train)
    
    # 6. 准备数据加载器
    print("\n步骤6: 准备数据加载器...")
    train_loader, val_loader, test_loader = trainer.prepare_data(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32
    )
    
    # 7. 训练模型
    print("\n步骤7: 训练模型...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=50,
        learning_rate=0.001,
        patience=10,
        class_weights=class_weights
    )
    
    # 8. 评估模型
    print("\n步骤8: 评估模型...")
    predictions, targets, probabilities = trainer.evaluate(test_loader)
    
    # 9. 可视化结果
    print("\n步骤9: 可视化结果...")
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    visualize_results(history, predictions, targets, probabilities, model_name=model_type)
    
    # 10. 保存模型
    print("\n步骤10: 保存模型...")
    trainer.save_model(f"models/best_{model_type.lower()}.pth")
    
    print("\n" + "=" * 60)
    print(f"{model_type} 训练流程完成！")
    print("=" * 60)
    
    return trainer, history, predictions, targets, probabilities


# ================ 直接运行主函数 ================
if __name__ == "__main__":
    # 直接运行主函数
    trainer, history, predictions, targets, probabilities = main()