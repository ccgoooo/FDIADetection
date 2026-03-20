"""
无监督深度自编码器（DAE）用于电力系统虚假数据注入攻击检测
仅使用正常样本训练，基于重构误差进行异常检测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(LSTM_AE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 编码器
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)

        # 解码器（使用 LSTMCell 方便逐时间步控制）
        self.decoder_cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 编码
        _, (h_n, c_n) = self.encoder(x)
        # 取最后一层的隐藏状态作为解码器初始状态
        h = h_n[-1]  # (batch, hidden_size)
        c = c_n[-1]  # (batch, hidden_size)

        # 解码（训练时使用 Teacher Forcing）
        outputs = []
        # 第一个解码器输入用零向量（或可学习嵌入，简单起见用零）
        dec_input = torch.zeros(batch_size, x.size(-1)).to(x.device)
        for t in range(seq_len):
            h, c = self.decoder_cell(dec_input, (h, c))
            h = self.dropout(h)
            out = self.fc(h)      # (batch, input_size)
            outputs.append(out.unsqueeze(1))
            # Teacher Forcing：下一个输入使用真实值（也可用 out.detach() 防止梯度传播）
            dec_input = x[:, t, :]   # 训练时使用真实值
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, input_size)
        return outputs

    def encode(self, x):
        # 如果需要获取编码向量，可返回最后一个隐藏状态
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]


class UnsupervisedDAE:
    """无监督深度自编码器攻击检测器"""
    
    def __init__(self, input_dim=84, hidden_size=64,num_layers=2, seq_len=24, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化无监督DAE
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.input_dim = input_dim
        self.norm_mean = None   # 新增：保存归一化均值
        self.norm_std = None    # 新增：保存归一化标准差
        self.seq_len = seq_len
        self.dae = LSTM_AE(input_size=input_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=0.2).to(device)
        # 创建自编码器
                
        # 损失函数（仅重构损失）
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.dae.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 阈值（用于检测攻击）
        self.threshold = None

    def fit_normalizer(self, X_train_normal):
        """
        使用训练集正常样本计算每个特征的均值和标准差（沿样本和窗口维度）
        X_train_normal: (n_samples, window_size, n_features)
        """
        # 合并样本和窗口维度，计算每个特征的均值和标准差
        all_data = X_train_normal.reshape(-1, X_train_normal.shape[-1])
        self.norm_mean = np.mean(all_data, axis=0)
        self.norm_std = np.std(all_data, axis=0) + 1e-8  # 加小常数避免除零
        print(f"归一化参数计算完成：mean shape {self.norm_mean.shape}, std shape {self.norm_std.shape}")
    
    def normalize(self, X):
        """
        对数据应用标准化（基于训练集统计量）
        X: (..., n_features) 任意形状，最后一维是特征
        直接处理三维输入
        """
        return (X - self.norm_mean) / self.norm_std

    def prepare_data(self, X_train_normal, X_val_normal, batch_size=32):
        """
        准备训练数据（仅正常样本）
        
        参数:
            X_train_normal: 训练正常数据 (n_samples, window_size, n_features)
            X_val_normal: 验证正常数据
            batch_size: 批次大小
        """
        # 直接转换为张量，不再取最后一个时间步
        X_train_tensor = torch.FloatTensor(X_train_normal).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_normal).to(self.device)

        # 创建数据集和加载器（与之前相同）
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.dae.train()
        
        epoch_loss = 0
        
        for batch_X, in train_loader:
            # 前向传播
            reconstructed = self.dae(batch_X)
            
            # 重构损失
            loss = self.criterion(reconstructed, batch_X)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dae.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """验证"""
        self.dae.eval()
        
        val_loss = 0
        all_reconstructions = []
        all_originals = []
        all_recon_errors = []
        
        with torch.no_grad():
            for batch_X, in val_loader:
                # 前向传播
                reconstructed = self.dae(batch_X)
                
                # 计算损失
                loss = self.criterion(reconstructed, batch_X)
                val_loss += loss.item()
                
                # 收集重构误差
                recon_error = torch.mean((reconstructed - batch_X) ** 2, dim=(1,2)).cpu().numpy()
                all_recon_errors.extend(recon_error)
                
                # 收集重构样本（用于可视化）
                if len(all_reconstructions) < 100:
                    all_reconstructions.extend(reconstructed.cpu().numpy())
                    all_originals.extend(batch_X.cpu().numpy())
        
        # 计算平均损失
        avg_loss = val_loss / len(val_loader)
        
        return avg_loss, np.array(all_recon_errors), all_reconstructions, all_originals
    
    def train(self, X_train_normal, X_val_normal, epochs=200, batch_size=64, patience=20):
        """
        训练无监督DAE
        
        参数:
            X_train_normal: 训练正常数据
            X_val_normal: 验证正常数据
            epochs: 训练轮数
            batch_size: 批次大小
            patience: 早停耐心值
        """
        print("开始训练无监督深度自编码器...")

        # 第一步：计算并保存归一化参数
        self.fit_normalizer(X_train_normal)

        # 第二步：对训练和验证数据归一化
        X_train_normal = self.normalize(X_train_normal)
        X_val_normal = self.normalize(X_val_normal)
        
        # 准备数据
        train_loader, val_loader = self.prepare_data(
            X_train_normal, X_val_normal, batch_size
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_errors, reconstructions, originals = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.dae.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停在 epoch {epoch+1}")
                    break
            
            # 每50个epoch可视化一次重构效果
            if (epoch + 1) % 50 == 0 and len(reconstructions) > 0:
                self._visualize_reconstruction(
                    originals[:5], reconstructions[:5], epoch + 1
                )
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.dae.load_state_dict(best_model_state)
            print("\n加载最佳模型")
        
        # 在验证集上计算阈值（基于正常样本）
        self._compute_threshold(X_val_normal)
        
        print("训练完成!")
        
        # 绘制训练历史
        self.plot_training_history()
        
        return self
    
    def _compute_threshold(self, X_val_normal, percentile=95):
        """
        X_val_normal: 已归一化的三维验证集正常样本
        """
        print(f"\n计算检测阈值（{percentile}% 分位数）...")
        X_val_tensor = torch.FloatTensor(X_val_normal).to(self.device)
        self.dae.eval()
        with torch.no_grad():
            reconstructed = self.dae(X_val_tensor)  # 输出 (batch, seq_len, n_features)
            # 计算每个样本的 MSE：对每个时间步和特征求平方，然后沿时间步和特征求平均
            recon_errors = torch.mean((reconstructed - X_val_tensor) ** 2, dim=(1,2)).cpu().numpy()
        self.threshold = np.percentile(recon_errors, percentile)
        print(f"重构误差统计：均值={np.mean(recon_errors):.6f}, 标准差={np.std(recon_errors):.6f}")
        print(f"阈值（{percentile}% 分位数）: {self.threshold:.6f}")
        return self.threshold
    
    def _visualize_reconstruction(self, originals, reconstructions, epoch):
        """
        originals, reconstructions: 列表，每个元素为 (seq_len, n_features) 的numpy数组
        """
        n_samples = min(3, len(originals))          # 展示3个样本
        seq_len = originals[0].shape[0]
        n_features = originals[0].shape[1]
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            orig = originals[i]      # (seq_len, n_features)
            recon = reconstructions[i]
            
            # 计算每个时间步的MSE（所有特征平均），用于子图标题
            step_mse = np.mean((orig - recon) ** 2, axis=1)   # (seq_len,)
            
            # 左侧子图：展示原始数据的热力图（便于观察整体模式）
            im0 = axes[i, 0].imshow(orig.T, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 0].set_title(f'样本 {i+1}: 原始数据 (特征按行)')
            axes[i, 0].set_xlabel('时间步')
            axes[i, 0].set_ylabel('特征索引')
            plt.colorbar(im0, ax=axes[i, 0])
            
            # 右侧子图：展示重构数据的热力图
            im1 = axes[i, 1].imshow(recon.T, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[i, 1].set_title(f'样本 {i+1}: 重构数据 (Epoch {epoch})')
            axes[i, 1].set_xlabel('时间步')
            axes[i, 1].set_ylabel('特征索引')
            plt.colorbar(im1, ax=axes[i, 1])
            
            # 也可以添加一条曲线：展示第一个特征的时间序列对比
            # axes[i, 2] 如果有更多子图可以添加
            
        plt.tight_layout()
        plt.savefig(f'figures/lstmae_reconstruction_epoch_{epoch}.png', dpi=150)
        plt.close()
    
    def plot_training_history(self, save_path="figures/lstmae_ervised_training_history.png"):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        plt.plot(epochs, self.train_history['train_loss'], 'b-', label='训练损失', alpha=0.7)
        plt.plot(epochs, self.train_history['val_loss'], 'r-', label='验证损失', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('损失 (MSE)')
        plt.title('无监督DAE训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练历史保存到 {save_path}")
        
        plt.show()
    
    def predict(self, X_test):
        if self.norm_mean is not None and self.norm_std is not None:
            X_test = self.normalize(X_test)
        self.dae.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            reconstructed = self.dae(X_test_tensor)
            recon_errors = torch.mean((reconstructed - X_test_tensor) ** 2, dim=(1,2)).cpu().numpy()
        if self.threshold is None:
            self.threshold = np.percentile(recon_errors, 95)
            print(f"警告: 未设置阈值，使用默认95%分位数: {self.threshold:.6f}")
        predictions = (recon_errors > self.threshold).astype(int)
        return predictions, recon_errors
    
    def evaluate(self, X_test, y_test):
        """
        在测试集上评估模型性能
        
        参数:
            X_test: 测试数据
            y_test: 真实标签 (0=正常, 1=攻击)
        """
        print("\n" + "="*60)
        print("在测试集上评估无监督DAE性能")
        print("="*60)
        
        predictions, recon_errors = self.predict(X_test)
        
        # 计算评估指标
        metrics = self._compute_evaluation_metrics(y_test, predictions, recon_errors)
        
        # 可视化结果
        self._visualize_evaluation_results(y_test, predictions, recon_errors)
        
        return predictions, recon_errors, metrics
    
    def _compute_evaluation_metrics(self, true_labels, predictions, recon_errors):
        """计算评估指标"""
        print("\n评估指标:")
        print("-"*40)
        
        # 基础指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        print(f"\n混淆矩阵:")
        print(f"         预测正常   预测攻击")
        print(f"实际正常  {cm[0, 0]:8d}  {cm[0, 1]:8d}")
        print(f"实际攻击  {cm[1, 0]:8d}  {cm[1, 1]:8d}")
        
        # 分类报告
        print(f"\n分类报告:")
        print(classification_report(true_labels, predictions, target_names=['正常', '攻击']))
        
        # 重构误差统计
        normal_errors = recon_errors[true_labels == 0]
        attack_errors = recon_errors[true_labels == 1] if np.sum(true_labels == 1) > 0 else np.array([])
        
        print(f"\n重构误差统计:")
        print(f"正常样本平均重构误差: {np.mean(normal_errors):.6f} ± {np.std(normal_errors):.6f}")
        if len(attack_errors) > 0:
            print(f"攻击样本平均重构误差: {np.mean(attack_errors):.6f} ± {np.std(attack_errors):.6f}")
            print(f"误差比率 (攻击/正常): {np.mean(attack_errors)/np.mean(normal_errors):.2f}")
        
        print(f"检测阈值: {self.threshold:.6f}")
        
        # 计算ROC曲线和AUC
        if np.sum(true_labels == 1) > 0 and np.sum(true_labels == 0) > 0:
            fpr, tpr, _ = roc_curve(true_labels, recon_errors)
            roc_auc = auc(fpr, tpr)
            print(f"\nROC曲线下面积 (AUC): {roc_auc:.4f}")
            
            # 计算精确率-召回率曲线
            precision_vals, recall_vals, _ = precision_recall_curve(true_labels, recon_errors)
            pr_auc = auc(recall_vals, precision_vals)
            print(f"PR曲线下面积: {pr_auc:.4f}")
        else:
            roc_auc, pr_auc = 0, 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'threshold': self.threshold,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def _visualize_evaluation_results(self, true_labels, predictions, recon_errors, save_dir="figures"):
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        
        # 1. 重构误差分布
        normal_errors = recon_errors[true_labels == 0]
        attack_errors = recon_errors[true_labels == 1] if np.sum(true_labels == 1) > 0 else np.array([])
        axes[0, 0].hist(normal_errors, bins=50, alpha=0.7, label='正常样本', density=True, color='blue')
        if len(attack_errors) > 0:
            axes[0, 0].hist(attack_errors, bins=50, alpha=0.7, label='攻击样本', density=True, color='red')
        axes[0, 0].axvline(self.threshold, color='green', linestyle='--', linewidth=2, label=f'阈值={self.threshold:.4f}')
        axes[0, 0].set_title('重构误差分布')
        axes[0, 0].set_xlabel('重构误差 (MSE)')
        axes[0, 0].set_ylabel('密度')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC曲线
        if np.sum(true_labels == 1) > 0 and np.sum(true_labels == 0) > 0:
            fpr, tpr, _ = roc_curve(true_labels, recon_errors)
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
            axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('假正率 (False Positive Rate)')
            axes[0, 1].set_ylabel('真正率 (True Positive Rate)')
            axes[0, 1].set_title('ROC曲线')
            axes[0, 1].legend(loc="lower right")
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PR曲线
        if np.sum(true_labels == 1) > 0:
            precision_vals, recall_vals, _ = precision_recall_curve(true_labels, recon_errors)
            pr_auc = auc(recall_vals, precision_vals)
            axes[1, 0].plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR曲线 (AUC = {pr_auc:.2f})')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('召回率 (Recall)')
            axes[1, 0].set_ylabel('精确率 (Precision)')
            axes[1, 0].set_title('精确率-召回率曲线')
            axes[1, 0].legend(loc="upper right")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('预测标签')
        axes[1, 1].set_ylabel('真实标签')
        axes[1, 1].set_title('混淆矩阵')
        
        # 5. 检测结果时间序列（取前200个样本）
        sample_size = min(200, len(true_labels))
        indices = np.arange(sample_size)
        axes[2, 0].plot(indices, recon_errors[:sample_size], 'b-', alpha=0.7, label='重构误差')
        axes[2, 0].axhline(y=self.threshold, color='r', linestyle='--', alpha=0.7, label='检测阈值')
        # 标记攻击区域
        attack_indices = np.where(true_labels[:sample_size] == 1)[0]
        if len(attack_indices) > 0:
            axes[2, 0].fill_between(indices, 0, np.max(recon_errors[:sample_size]), 
                                    where=np.isin(indices, attack_indices),
                                    color='red', alpha=0.2, label='真实攻击')
        # 标记检测到的攻击
        detected_indices = np.where(predictions[:sample_size] == 1)[0]
        for idx in detected_indices:
            axes[2, 0].plot(idx, recon_errors[idx], 'ro', markersize=4, alpha=0.5)
        axes[2, 0].set_xlabel('样本索引')
        axes[2, 0].set_ylabel('重构误差')
        axes[2, 0].set_title('检测结果（前200个样本）')
        axes[2, 0].legend(loc='upper right')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 误差对比箱线图
        if len(attack_errors) > 0:
            error_data = [normal_errors, attack_errors]
            axes[2, 1].boxplot(error_data, labels=['正常', '攻击'])
            axes[2, 1].set_title('重构误差对比（箱线图）')
            axes[2, 1].set_ylabel('重构误差')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, '无攻击样本数据', 
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=axes[2, 1].transAxes,
                            fontsize=12)
            axes[2, 1].set_title('重构误差对比')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/lstmae_unsupervised_evaluation.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path="models/dae_unsupervised"):
        """保存模型"""
        import os
        import json
        os.makedirs(path, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'dae_state_dict': self.dae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'hidden_size': self.dae.hidden_size,      # 新增
            'num_layers': self.dae.num_layers,        # 新增
            'seq_len': self.seq_len,
            'norm_mean': self.norm_mean,
            'norm_std': self.norm_std,
            'train_history': self.train_history
        }, f"{path}/model.pth")
        
        # 保存模型配置
        config = {
            'input_dim': self.input_dim,
            'hidden_size': self.dae.hidden_size,      # 新增
            'num_layers': self.dae.num_layers,        # 新增
            'seq_len': self.seq_len,
            'device': str(self.device),
            'threshold': float(self.threshold) if self.threshold is not None else 0.0
        }
        
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"模型保存到 {path}")
    
    def load_model(self, path="models/dae_unsupervised"):
        checkpoint = torch.load(f"{path}/model.pth", map_location=self.device)
        
        # 读取超参数
        input_dim = checkpoint['input_dim']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        seq_len = checkpoint['seq_len']
        
        # 重新创建模型（确保结构一致）
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dae = LSTM_AE(input_size=input_dim, hidden_size=hidden_size,
                        num_layers=num_layers, dropout=0.2).to(self.device)
        
        # 加载模型权重
        self.dae.load_state_dict(checkpoint['dae_state_dict'])
        
        # 重新创建优化器（因为模型参数变了）
        self.optimizer = optim.Adam(self.dae.parameters(), lr=0.001, weight_decay=1e-5)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复其他属性
        self.threshold = checkpoint['threshold']
        self.norm_mean = checkpoint.get('norm_mean', None)
        self.norm_std = checkpoint.get('norm_std', None)
        self.train_history = checkpoint['train_history']
        
        print(f"模型从 {path} 加载，结构: input_dim={input_dim}, hidden_size={hidden_size}, num_layers={num_layers}, seq_len={seq_len}")
        print(f"阈值: {self.threshold}")


def analyze_false_negatives(detector, X_test, y_test):
    """
    分析漏报的攻击样本特征
    """
    predictions, recon_errors = detector.predict(X_test)
    
    # 找出漏报的攻击样本（真实攻击但未检测到）
    false_negatives = np.where((y_test == 1) & (predictions == 0))[0]
    
    print("\n" + "="*60)
    print(f"漏报攻击样本分析 (阈值={detector.threshold:.6f})")
    print("="*60)
    
    total_attacks = np.sum(y_test == 1)
    print(f"总攻击样本: {total_attacks}")
    print(f"漏报数量: {len(false_negatives)}")
    if total_attacks > 0:
        print(f"漏报率: {len(false_negatives)/total_attacks*100:.2f}%")
    
    if len(false_negatives) > 0:
        # 分析漏报样本的重构误差
        fn_errors = recon_errors[false_negatives]
        
        print(f"\n漏报样本重构误差统计:")
        print(f"  均值: {np.mean(fn_errors):.6f}")
        print(f"  中位数: {np.median(fn_errors):.6f}")
        print(f"  标准差: {np.std(fn_errors):.6f}")
        print(f"  最小值: {np.min(fn_errors):.6f}")
        print(f"  最大值: {np.max(fn_errors):.6f}")
        
        # 与正常样本对比
        normal_errors = recon_errors[y_test == 0]
        print(f"\n正常样本重构误差统计:")
        print(f"  均值: {np.mean(normal_errors):.6f}")
        print(f"  中位数: {np.median(normal_errors):.6f}")
        print(f"  标准差: {np.std(normal_errors):.6f}")
        print(f"  99%分位数: {np.percentile(normal_errors, 99):.6f}")
        
        # 可视化漏报样本
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(normal_errors, bins=50, alpha=0.7, label='正常样本', density=True, color='blue')
        plt.hist(fn_errors, bins=30, alpha=0.7, label='漏报攻击', density=True, color='red')
        plt.axvline(x=detector.threshold, color='green', linestyle='--', linewidth=2, label=f'阈值={detector.threshold:.4f}')
        plt.xlabel('重构误差')
        plt.ylabel('密度')
        plt.title('漏报攻击 vs 正常样本分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 箱线图对比
        data_to_plot = [normal_errors, fn_errors]
        plt.boxplot(data_to_plot, labels=['正常样本', '漏报攻击'])
        plt.axhline(y=detector.threshold, color='green', linestyle='--', linewidth=2, label=f'阈值={detector.threshold:.4f}')
        plt.ylabel('重构误差')
        plt.title('重构误差对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/false_negatives_analysis.png', dpi=150)
        plt.show()
        
        # 建议
        if np.max(fn_errors) < np.percentile(normal_errors, 95):
            print("误差小于百分位数（95）")
            print("\n⚠️ 警告：部分攻击样本的误差完全在正常范围内")
            print("建议：考虑使用更复杂的特征或时序信息")
        elif np.mean(fn_errors) < np.percentile(normal_errors, 90):
            print("误差大于百分位数（95）")
            print("\n📊 观察：漏报攻击的误差接近正常样本上限")
            print("建议：可以尝试稍微降低阈值")
    
    return false_negatives


def run_unsupervised_dae_detection():
    """运行无监督DAE攻击检测主流程"""
    
    print("="*60)
    print("无监督深度自编码器（DAE）攻击检测系统")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    file_path = "processed_data/unsupervised/physical_constrained_84"   # single_point_84、multi_point_84、slow_drift_84、physical_constrained_84
    try:
        X_train = np.load(os.path.join(file_path, "x_train.npy"))
        y_train = np.load(os.path.join(file_path, "y_train.npy"))
        X_val = np.load(os.path.join(file_path, "x_val.npy"))
        y_val = np.load(os.path.join(file_path, "y_val.npy"))
        X_test = np.load(os.path.join(file_path, "x_test.npy"))
        y_test = np.load(os.path.join(file_path, "y_test.npy"))
        
        print(f"训练数据: {X_train.shape}, {y_train.shape}")
        print(f"验证数据: {X_val.shape}, {y_val.shape}")
        print(f"测试数据: {X_test.shape}, {y_test.shape}")
        
        # 检查特征维度
        feature_dim = X_train.shape[2]
        print(f"特征维度: {feature_dim}")
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("请确保已运行数据生成管道")
        return
    
    # 2. 提取正常样本用于训练（y==0）
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]
    
    print(f"\n训练正常样本数: {len(X_train_normal)}")
    print(f"验证正常样本数: {len(X_val_normal)}")
    
    # 3. 创建无监督DAE模型
    print("\n2. 创建无监督DAE模型...")
    seq_len = X_train.shape[1]   # 窗口长度
    dae_detector = UnsupervisedDAE(
        input_dim=feature_dim,
        hidden_size=64,           # 可根据需要调整
        num_layers=2,
        seq_len=seq_len,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n3. 训练无监督DAE模型...")
    dae_detector.train(X_train_normal, X_val_normal, epochs=150, batch_size=128, patience=15)
      
    # 5. 在测试集上评估（包含攻击样本）
    print("\n4. 在测试集上评估...")
    predictions, recon_errors, metrics = dae_detector.evaluate(X_test, y_test)
    
    # 6. 分析漏报情况
    print("\n5. 分析漏报情况...")
    false_negatives = analyze_false_negatives(dae_detector, X_test, y_test)
    
    # 7. 保存模型
    print("\n6. 保存模型...")
    dae_detector.save_model("models/dae_unsupervised_final")
    
    # 8. 生成检测报告
    generate_detection_report(y_test, predictions, recon_errors, metrics)
    
    return dae_detector


def generate_detection_report(true_labels, predictions, recon_errors, metrics):
    """生成检测报告"""
    report = f"""
    =============================================
    无监督深度自编码器（DAE）攻击检测报告
    =============================================
    
    1. 总体性能指标
       - 准确率: {metrics['accuracy']:.4f}
       - 精确率: {metrics['precision']:.4f}
       - 召回率: {metrics['recall']:.4f}
       - F1分数: {metrics['f1']:.4f}
       - ROC AUC: {metrics['roc_auc']:.4f}
       - PR AUC: {metrics['pr_auc']:.4f}
    
    2. 检测结果统计
       - 总样本数: {len(true_labels)}
       - 真实攻击数: {np.sum(true_labels == 1)}
       - 检测攻击数: {np.sum(predictions == 1)}
       - 漏报数: {metrics['confusion_matrix'][1, 0] if metrics['confusion_matrix'].shape[0] > 1 else 0}
       - 误报数: {metrics['confusion_matrix'][0, 1] if metrics['confusion_matrix'].shape[0] > 1 else 0}
    
    3. 重构误差分析
       - 检测阈值: {metrics['threshold']:.6f}
       - 正常样本平均误差: {np.mean(recon_errors[true_labels == 0]):.6f}
       - 攻击样本平均误差: {np.mean(recon_errors[true_labels == 1]) if np.sum(true_labels == 1) > 0 else 0:.6f}
    """
    
    # 保存报告
    with open("models/dae_unsupervised_final/detection_report.txt", "w") as f:
        f.write(report)
    
    print(report)
    print("\n详细报告已保存到: models/dae_unsupervised_final/detection_report.txt")


def test_pretrained_model():
    """测试预训练的无监督模型"""
    print("加载预训练模型进行测试...")
    
    # 加载数据
    X_test = np.load("processed_data/X_test.npy")
    y_test = np.load("processed_data/y_test.npy")
    
    # 获取特征维度
    feature_dim = X_test.shape[2]
    
    # 创建模型
    dae_detector = UnsupervisedDAE(
        input_dim=feature_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 加载预训练模型
    dae_detector.load_model("models/dae_unsupervised_final")
    
    # 评估
    predictions, recon_errors, metrics = dae_detector.evaluate(X_test, y_test)
    
    return dae_detector


if __name__ == "__main__":
    # 运行无监督DAE攻击检测流程
    detector = run_unsupervised_dae_detection()
    
    # 或者测试预训练模型
    # detector = test_pretrained_model()