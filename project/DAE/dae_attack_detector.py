# dae_attack_detector.py
"""
深度自编码器（DAE）用于电力系统虚假数据注入攻击检测
实现半监督学习，结合少量标签数据进行攻击检测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DAE(nn.Module):
    """深度自编码器网络"""
    
    def __init__(self, input_dim=84, hidden_dims=[64, 32, 16], dropout_rate=0.2):
        """
        初始化自编码器
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表（对称结构）
            dropout_rate: Dropout率
        """
        super(DAE, self).__init__()
        
        # 编码器部分
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            encoder_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 瓶颈层
        self.bottleneck_dim = hidden_dims[-1]
        
        # 解码器部分（对称结构）
        decoder_layers = []
        hidden_dims_reverse = hidden_dims[::-1]  # 反转列表
        
        for i, hidden_dim in enumerate(hidden_dims_reverse):
            if i == len(hidden_dims_reverse) - 1:
                # 最后一层输出到原始维度
                decoder_layers.append(nn.Linear(current_dim, input_dim))
            else:
                decoder_layers.append(nn.Linear(current_dim, hidden_dim))
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
                decoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.Dropout(dropout_rate))
                current_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """前向传播"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        """仅编码"""
        return self.encoder(x)


class SemiSupervisedDAE:
    """半监督深度自编码器攻击检测器"""
    
    def __init__(self, input_dim=84, hidden_dims=[64, 32, 16], device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化半监督DAE
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.input_dim = input_dim
        
        # 创建自编码器
        self.dae = DAE(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
        
        # 创建分类头（用于半监督学习）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 二分类输出概率
        ).to(device)
        
        # 损失函数
        self.reconstruction_criterion = nn.MSELoss()
        self.classification_criterion = nn.BCELoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.dae.parameters()) + list(self.classifier.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # 训练历史
        self.train_history = {
            'reconstruction_loss': [],
            'classification_loss': [],
            'total_loss': [],
            'val_reconstruction_loss': [],
            'val_classification_loss': [],
            'val_total_loss': []
        }
        
        # 阈值（用于检测攻击）
        self.threshold = None
        
    def prepare_data(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        准备数据，处理标签不平衡问题
        
        参数:
            X_train: 训练数据 (n_samples, window_size, n_features)
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            batch_size: 批次大小
        """
        print("准备数据...")
        
        # 1. 将窗口数据展平或取最后一个时间步
        # 这里我们取窗口最后一个时间步，因为攻击通常在当前时刻
        X_train_flat = X_train[:, -1, :]  # 取最后一个时间步
        X_val_flat = X_val[:, -1, :]
        
        print(f"训练数据形状: {X_train.shape} -> {X_train_flat.shape}")
        print(f"验证数据形状: {X_val.shape} -> {X_val_flat.shape}")
        
        # 2. 统计类别分布
        train_normal_idx = np.where(y_train == 0)[0]
        train_attack_idx = np.where(y_train == 1)[0]
        
        print(f"\n训练集类别分布:")
        print(f"  正常样本: {len(train_normal_idx)} ({len(train_normal_idx)/len(y_train)*100:.1f}%)")
        print(f"  攻击样本: {len(train_attack_idx)} ({len(train_attack_idx)/len(y_train)*100:.1f}%)")
        
        # 3. 处理类别不平衡：对正常样本降采样，对攻击样本过采样
        # 目标：使正常:攻击 ≈ 3:1
        target_ratio = 3.0
        
        if len(train_attack_idx) > 0:
            # 计算需要的正常样本数量
            target_normal_count = int(len(train_attack_idx) * target_ratio)
            
            if len(train_normal_idx) > target_normal_count:
                # 正常样本降采样
                selected_normal_idx = np.random.choice(
                    train_normal_idx, 
                    size=min(target_normal_count, len(train_normal_idx)), 
                    replace=False
                )
            else:
                selected_normal_idx = train_normal_idx
            
            # 合并索引
            selected_idx = np.concatenate([selected_normal_idx, train_attack_idx])
            
            # 打乱顺序
            np.random.shuffle(selected_idx)
            
            X_train_balanced = X_train_flat[selected_idx]
            y_train_balanced = y_train[selected_idx]
        else:
            X_train_balanced = X_train_flat
            y_train_balanced = y_train
        
        print(f"\n平衡后训练集:")
        print(f"  总样本数: {len(X_train_balanced)}")
        normal_count = np.sum(y_train_balanced == 0)
        attack_count = np.sum(y_train_balanced == 1)
        print(f"  正常: {normal_count} ({normal_count/len(y_train_balanced)*100:.1f}%)")
        print(f"  攻击: {attack_count} ({attack_count/len(y_train_balanced)*100:.1f}%)")
        
        # 4. 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_balanced).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_balanced).to(self.device).unsqueeze(1)
        
        X_val_tensor = torch.FloatTensor(X_val_flat).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device).unsqueeze(1)
        
        # 5. 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, X_train_flat, X_val_flat
    
    def train_epoch(self, train_loader, lambda_cls=0.5):
        """训练一个epoch"""
        self.dae.train()
        self.classifier.train()
        
        epoch_recon_loss = 0
        epoch_cls_loss = 0
        epoch_total_loss = 0
        
        for batch_X, batch_y in train_loader:
            # 前向传播
            reconstructed, encoded = self.dae(batch_X)
            
            # 重构损失（所有样本）
            recon_loss = self.reconstruction_criterion(reconstructed, batch_X)
            
            # 分类损失（仅对有标签样本）
            # 注意：在我们的设置中，所有样本都有标签，但我们可以模拟半监督
            classification_output = self.classifier(encoded)
            cls_loss = self.classification_criterion(classification_output, batch_y)
            
            # 总损失
            total_loss = recon_loss + lambda_cls * cls_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dae.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录损失
            epoch_recon_loss += recon_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_total_loss += total_loss.item()
        
        # 计算平均损失
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)
        
        return avg_recon_loss, avg_cls_loss, avg_total_loss
    
    def validate(self, val_loader, lambda_cls=0.5):
        """验证"""
        self.dae.eval()
        self.classifier.eval()
        
        val_recon_loss = 0
        val_cls_loss = 0
        val_total_loss = 0
        
        all_reconstructions = []
        all_originals = []
        all_labels = []
        all_recon_errors = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # 前向传播
                reconstructed, encoded = self.dae(batch_X)
                
                # 计算损失
                recon_loss = self.reconstruction_criterion(reconstructed, batch_X)
                classification_output = self.classifier(encoded)
                cls_loss = self.classification_criterion(classification_output, batch_y)
                total_loss = recon_loss + lambda_cls * cls_loss
                
                # 记录损失
                val_recon_loss += recon_loss.item()
                val_cls_loss += cls_loss.item()
                val_total_loss += total_loss.item()
                
                # 收集重构误差
                recon_error = torch.mean((reconstructed - batch_X) ** 2, dim=1).cpu().numpy()
                all_recon_errors.extend(recon_error)
                all_labels.extend(batch_y.cpu().numpy().flatten())
                
                # 收集重构样本（用于可视化）
                if len(all_reconstructions) < 100:  # 只保存前100个用于可视化
                    all_reconstructions.extend(reconstructed.cpu().numpy())
                    all_originals.extend(batch_X.cpu().numpy())
        
        # 计算平均损失
        avg_recon_loss = val_recon_loss / len(val_loader)
        avg_cls_loss = val_cls_loss / len(val_loader)
        avg_total_loss = val_total_loss / len(val_loader)
        
        # 计算验证指标
        val_metrics = self._compute_validation_metrics(
            np.array(all_recon_errors), 
            np.array(all_labels)
        )
        
        return avg_recon_loss, avg_cls_loss, avg_total_loss, val_metrics, all_reconstructions, all_originals
    
    def _compute_validation_metrics(self, recon_errors, labels):
        """计算验证指标"""
        metrics = {}
        
        # 分离正常和攻击的重构误差
        normal_errors = recon_errors[labels == 0]
        attack_errors = recon_errors[labels == 1] if len(labels[labels == 1]) > 0 else np.array([])
        
        # 基本统计
        metrics['normal_error_mean'] = np.mean(normal_errors) if len(normal_errors) > 0 else 0
        metrics['normal_error_std'] = np.std(normal_errors) if len(normal_errors) > 0 else 0
        metrics['attack_error_mean'] = np.mean(attack_errors) if len(attack_errors) > 0 else 0
        metrics['attack_error_std'] = np.std(attack_errors) if len(attack_errors) > 0 else 0
        
        # 计算最佳阈值（基于验证集）
        if len(attack_errors) > 0:
            # 使用F1分数优化阈值
            best_threshold, best_f1 = self._find_optimal_threshold(recon_errors, labels)
            metrics['best_threshold'] = best_threshold
            metrics['best_f1'] = best_f1
            
            # 使用该阈值计算分类指标
            predictions = (recon_errors > best_threshold).astype(int)
            metrics['accuracy'] = accuracy_score(labels, predictions)
            metrics['precision'] = precision_score(labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(labels, predictions, zero_division=0)
            metrics['f1'] = f1_score(labels, predictions, zero_division=0)
        
        return metrics
    
    def _find_optimal_threshold(self, recon_errors, labels):
        """寻找最佳阈值（最大化F1分数）"""
        # 生成候选阈值
        min_error = np.min(recon_errors)
        max_error = np.max(recon_errors)
        
        # 如果攻击样本太少，使用统计方法
        if np.sum(labels == 1) < 10:
            normal_errors = recon_errors[labels == 0]
            threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
            return threshold, 0
        
        thresholds = np.linspace(min_error, max_error, 100)
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = (recon_errors > threshold).astype(int)
            
            # 确保有正负样本
            if np.sum(predictions) == 0 or np.sum(predictions) == len(predictions):
                continue
            
            f1 = f1_score(labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200, 
              batch_size=64, lambda_cls=0.5, patience=20):
        """
        训练半监督DAE
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            lambda_cls: 分类损失权重
            patience: 早停耐心值
        """
        print("开始训练半监督深度自编码器...")
        
        # 准备数据
        train_loader, val_loader, X_train_flat, X_val_flat = self.prepare_data(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # 保存数据供后续使用
        self.X_train_flat = X_train_flat
        self.X_val_flat = X_val_flat
        self.y_train = y_train
        self.y_val = y_val
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            train_recon_loss, train_cls_loss, train_total_loss = self.train_epoch(
                train_loader, lambda_cls
            )
            
            # 验证
            val_recon_loss, val_cls_loss, val_total_loss, val_metrics, reconstructions, originals = self.validate(
                val_loader, lambda_cls
            )
            
            # 更新学习率
            self.scheduler.step(val_total_loss)
            
            # 记录历史
            self.train_history['reconstruction_loss'].append(train_recon_loss)
            self.train_history['classification_loss'].append(train_cls_loss)
            self.train_history['total_loss'].append(train_total_loss)
            self.train_history['val_reconstruction_loss'].append(val_recon_loss)
            self.train_history['val_classification_loss'].append(val_cls_loss)
            self.train_history['val_total_loss'].append(val_total_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  训练 - 重构: {train_recon_loss:.4f}, 分类: {train_cls_loss:.4f}, 总: {train_total_loss:.4f}")
                print(f"  验证 - 重构: {val_recon_loss:.4f}, 分类: {val_cls_loss:.4f}, 总: {val_total_loss:.4f}")
                
                if 'accuracy' in val_metrics:
                    print(f"  验证指标 - Acc: {val_metrics['accuracy']:.4f}, "
                          f"Prec: {val_metrics['precision']:.4f}, "
                          f"Rec: {val_metrics['recall']:.4f}, "
                          f"F1: {val_metrics['f1']:.4f}")
            
            # 早停检查
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                best_model_state = {
                    'dae': self.dae.state_dict(),
                    'classifier': self.classifier.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'threshold': val_metrics.get('best_threshold', 0.01)
                }
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
            self.dae.load_state_dict(best_model_state['dae'])
            self.classifier.load_state_dict(best_model_state['classifier'])
            self.threshold = best_model_state['threshold']
            print(f"\n加载最佳模型，阈值: {self.threshold:.6f}")
        
        print("训练完成!")
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 在验证集上评估最终性能
        self.evaluate_on_validation_set()
        
        return self
    
    def _visualize_reconstruction(self, originals, reconstructions, epoch):
        """可视化重构效果"""
        n_samples = min(5, len(originals))
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # 原始数据
            axes[i, 0].plot(originals[i][:20], 'b-', alpha=0.7, label='原始')
            axes[i, 0].set_title(f'样本 {i+1}: 原始数据')
            axes[i, 0].set_xlabel('特征索引')
            axes[i, 0].set_ylabel('值')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            
            # 重构数据
            axes[i, 1].plot(originals[i][:20], 'b-', alpha=0.5, label='原始')
            axes[i, 1].plot(reconstructions[i][:20], 'r--', alpha=0.7, label='重构')
            axes[i, 1].set_title(f'样本 {i+1}: 重构对比 (Epoch {epoch})')
            axes[i, 1].set_xlabel('特征索引')
            axes[i, 1].set_ylabel('值')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
            
            # 计算该样本的重构误差
            mse = np.mean((originals[i] - reconstructions[i]) ** 2)
            axes[i, 1].text(0.05, 0.95, f'MSE: {mse:.4f}', 
                           transform=axes[i, 1].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'figures/dae_reconstruction_epoch_{epoch}.png', dpi=150)
        plt.close()
    
    def plot_training_history(self, save_path="figures/dae_training_history.png"):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.train_history['total_loss']) + 1)
        
        # 1. 总损失
        axes[0, 0].plot(epochs, self.train_history['total_loss'], 'b-', label='训练总损失', alpha=0.7)
        axes[0, 0].plot(epochs, self.train_history['val_total_loss'], 'r-', label='验证总损失', alpha=0.7)
        axes[0, 0].set_title('总损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 重构损失
        axes[0, 1].plot(epochs, self.train_history['reconstruction_loss'], 'b-', label='训练重构损失', alpha=0.7)
        axes[0, 1].plot(epochs, self.train_history['val_reconstruction_loss'], 'r-', label='验证重构损失', alpha=0.7)
        axes[0, 1].set_title('重构损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 分类损失
        axes[1, 0].plot(epochs, self.train_history['classification_loss'], 'b-', label='训练分类损失', alpha=0.7)
        axes[1, 0].plot(epochs, self.train_history['val_classification_loss'], 'r-', label='验证分类损失', alpha=0.7)
        axes[1, 0].set_title('分类损失')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 损失对比（对数尺度）
        axes[1, 1].semilogy(epochs, self.train_history['total_loss'], 'b-', label='训练总损失', alpha=0.7)
        axes[1, 1].semilogy(epochs, self.train_history['val_total_loss'], 'r-', label='验证总损失', alpha=0.7)
        axes[1, 1].semilogy(epochs, self.train_history['reconstruction_loss'], 'g-', label='训练重构损失', alpha=0.5)
        axes[1, 1].semilogy(epochs, self.train_history['val_reconstruction_loss'], 'y-', label='验证重构损失', alpha=0.5)
        axes[1, 1].set_title('损失对比（对数尺度）')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('损失 (log)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练历史保存到 {save_path}")
        
        plt.show()
    
    def evaluate_on_validation_set(self):
        """在验证集上评估模型性能"""
        print("\n" + "="*60)
        print("在验证集上评估DAE性能")
        print("="*60)
        
        self.dae.eval()
        self.classifier.eval()
        
        # 将数据移到设备
        X_val_tensor = torch.FloatTensor(self.X_val_flat).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val).to(self.device)
        
        with torch.no_grad():
            # 计算重构误差
            reconstructed, encoded = self.dae(X_val_tensor)
            recon_errors = torch.mean((reconstructed - X_val_tensor) ** 2, dim=1).cpu().numpy()
            
            # 分类器输出
            classifier_output = self.classifier(encoded).cpu().numpy().flatten()
        
        # 使用重构误差进行攻击检测
        if self.threshold is None:
            # 如果没有阈值，计算一个
            normal_errors = recon_errors[self.y_val == 0]
            self.threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
        
        predictions = (recon_errors > self.threshold).astype(int)
        
        # 计算评估指标
        self._compute_evaluation_metrics(self.y_val, predictions, recon_errors, classifier_output)
        
        # 可视化结果
        self._visualize_evaluation_results(self.y_val, predictions, recon_errors)
        
        return predictions, recon_errors
    
    def _compute_evaluation_metrics(self, true_labels, predictions, recon_errors, classifier_scores=None):
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'threshold': self.threshold,
            'roc_auc': roc_auc if 'roc_auc' in locals() else 0,
            'pr_auc': pr_auc if 'pr_auc' in locals() else 0
        }
    
    def _visualize_evaluation_results(self, true_labels, predictions, recon_errors, save_dir="figures"):
        """可视化评估结果"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        # 3. 精确率-召回率曲线
        if np.sum(true_labels == 1) > 0:
            precision_vals, recall_vals, _ = precision_recall_curve(true_labels, recon_errors)
            pr_auc = auc(recall_vals, precision_vals)
            
            axes[0, 2].plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR曲线 (AUC = {pr_auc:.2f})')
            axes[0, 2].set_xlim([0.0, 1.0])
            axes[0, 2].set_ylim([0.0, 1.05])
            axes[0, 2].set_xlabel('召回率 (Recall)')
            axes[0, 2].set_ylabel('精确率 (Precision)')
            axes[0, 2].set_title('精确率-召回率曲线')
            axes[0, 2].legend(loc="upper right")
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 混淆矩阵热力图
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('预测标签')
        axes[1, 0].set_ylabel('真实标签')
        axes[1, 0].set_title('混淆矩阵')
        
        # 5. 检测结果时间序列（取前200个样本）
        sample_size = min(200, len(true_labels))
        indices = range(sample_size)
        
        axes[1, 1].plot(indices, recon_errors[:sample_size], 'b-', alpha=0.7, label='重构误差')
        axes[1, 1].axhline(y=self.threshold, color='r', linestyle='--', alpha=0.7, label='检测阈值')
        
        # 标记攻击区域
        attack_indices = np.where(true_labels[:sample_size] == 1)[0]
        if len(attack_indices) > 0:
            axes[1, 1].fill_between(indices, 0, np.max(recon_errors[:sample_size]), 
                                   where=np.isin(indices, attack_indices),
                                   color='red', alpha=0.2, label='真实攻击')
        
        # 标记检测到的攻击
        detected_indices = np.where(predictions[:sample_size] == 1)[0]
        for idx in detected_indices:
            axes[1, 1].plot(idx, recon_errors[idx], 'ro', markersize=4, alpha=0.5)
        
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('重构误差')
        axes[1, 1].set_title('检测结果（前200个样本）')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 误差对比箱线图
        if len(attack_errors) > 0:
            error_data = [normal_errors, attack_errors]
            axes[1, 2].boxplot(error_data, labels=['正常', '攻击'])
            axes[1, 2].set_title('重构误差对比（箱线图）')
            axes[1, 2].set_ylabel('重构误差')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, '无攻击样本数据', 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axes[1, 2].transAxes,
                           fontsize=12)
            axes[1, 2].set_title('重构误差对比')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dae_evaluation_results.png", dpi=150, bbox_inches='tight')
        print(f"\n评估结果可视化保存到 {save_dir}/dae_evaluation_results.png")
        plt.show()
    
    def predict(self, X_test):
        """
        预测新数据
        
        参数:
            X_test: 测试数据 (n_samples, window_size, n_features)
        
        返回:
            predictions: 预测标签 (1=攻击, 0=正常)
            recon_errors: 重构误差
            confidence: 置信度分数
        """
        print("进行攻击检测预测...")
        
        # 确保模型在评估模式
        self.dae.eval()
        self.classifier.eval()
        
        # 处理输入数据（取最后一个时间步）
        if len(X_test.shape) == 3:
            X_test_flat = X_test[:, -1, :]
        else:
            X_test_flat = X_test
        
        # 转换为张量
        X_test_tensor = torch.FloatTensor(X_test_flat).to(self.device)
        
        with torch.no_grad():
            # 计算重构误差
            reconstructed, encoded = self.dae(X_test_tensor)
            recon_errors = torch.mean((reconstructed - X_test_tensor) ** 2, dim=1).cpu().numpy()
            
            # 分类器输出（作为置信度）
            classifier_output = self.classifier(encoded).cpu().numpy().flatten()
        
        # 基于重构误差进行预测
        if self.threshold is None:
            # 如果没有阈值，设置一个默认值
            self.threshold = np.percentile(recon_errors, 95)  # 使用95百分位数
        
        predictions = (recon_errors > self.threshold).astype(int)
        
        print(f"预测完成: {len(predictions)} 个样本")
        print(f"预测攻击比例: {np.sum(predictions)/len(predictions)*100:.2f}%")
        
        return predictions, recon_errors, classifier_output
    
    def save_model(self, path="models/dae_attack_detector"):
        """保存模型"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'dae_state_dict': self.dae.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'train_history': self.train_history
        }, f"{path}/model.pth")
        
        # 保存模型配置
        import json
        config = {
            'input_dim': self.input_dim,
            'device': str(self.device),
            'threshold': float(self.threshold) if self.threshold is not None else 0.0
        }
        
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"模型保存到 {path}")
    
    def load_model(self, path="models/dae_attack_detector"):
        """加载模型"""
        checkpoint = torch.load(f"{path}/model.pth", map_location=self.device)
        
        self.dae.load_state_dict(checkpoint['dae_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.threshold = checkpoint['threshold']
        self.train_history = checkpoint['train_history']
        
        print(f"模型从 {path} 加载")
        print(f"阈值: {self.threshold}")


def run_dae_attack_detection():
    """运行DAE攻击检测主流程"""
    
    print("="*60)
    print("深度自编码器（DAE）攻击检测系统")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    try:
        X_train = np.load("processed_data/X_train.npy")
        y_train = np.load("processed_data/y_train.npy")
        X_val = np.load("processed_data/X_val.npy")
        y_val = np.load("processed_data/y_val.npy")
        X_test = np.load("processed_data/X_test.npy")
        y_test = np.load("processed_data/y_test.npy")
        
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
    
    # 2. 创建并训练DAE模型
    print("\n2. 创建DAE模型...")
    dae_detector = SemiSupervisedDAE(
        input_dim=feature_dim,
        hidden_dims=[64, 32, 16],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 3. 训练模型（如果已有训练好的模型，可以跳过这一步）
    print("\n3. 训练DAE模型...")
    dae_detector.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=150,  # 可以根据需要调整
        batch_size=128,
        lambda_cls=0.3,  # 分类损失权重
        patience=15
    )
    
    # 4. 在测试集上评估
    print("\n4. 在测试集上评估...")
    test_predictions, test_errors, test_confidence = dae_detector.predict(X_test)
    
    # 计算测试集指标
    test_metrics = dae_detector._compute_evaluation_metrics(y_test, test_predictions, test_errors)
    
    # 5. 保存模型
    print("\n5. 保存模型...")
    dae_detector.save_model("models/dae_attack_detector_final")
    
    # 6. 生成检测报告
    generate_detection_report(y_test, test_predictions, test_errors, test_metrics)
    
    return dae_detector


def generate_detection_report(true_labels, predictions, recon_errors, metrics):
    """生成检测报告"""
    report = f"""
    =============================================
    深度自编码器（DAE）攻击检测报告
    =============================================
    
    1. 总体性能指标
       - 准确率: {metrics['accuracy']:.4f}
       - 精确率: {metrics['precision']:.4f}
       - 召回率: {metrics['recall']:.4f}
       - F1分数: {metrics['f1']:.4f}
       - ROC AUC: {metrics.get('roc_auc', 0):.4f}
       - PR AUC: {metrics.get('pr_auc', 0):.4f}
    
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
    
    4. 评估结论
    """
    
    # 添加结论
    if metrics['f1'] > 0.8:
        report += "    - 检测性能优秀 (F1 > 0.8)\n"
    elif metrics['f1'] > 0.6:
        report += "    - 检测性能良好 (F1 > 0.6)\n"
    else:
        report += "    - 检测性能有待改进\n"
    
    if metrics.get('roc_auc', 0) > 0.9:
        report += "    - 分类器区分能力强 (AUC > 0.9)\n"
    
    # 检测延迟分析（对于时间序列）
    report += f"""
    5. 实际应用建议
       - 阈值设置: {metrics['threshold']:.6f}
       - 可根据实际需求调整阈值（提高召回率或精确率）
       - 建议定期重新训练以适应系统变化
    """
    
    # 保存报告
    with open("model/dae_attack_detector_final/dae_detection_report.txt", "w") as f:
        f.write(report)
    
    print(report)
    print("\n详细报告已保存到: model/dae_attack_detector_final/dae_detection_report.txt")


def test_pretrained_model():
    """测试预训练模型"""
    print("加载预训练模型进行测试...")
    
    # 加载数据
    X_test = np.load("processed_data/X_test.npy")
    y_test = np.load("processed_data/y_test.npy")
    
    # 获取特征维度
    feature_dim = X_test.shape[2]
    
    # 创建模型
    dae_detector = SemiSupervisedDAE(
        input_dim=feature_dim,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 加载预训练模型
    dae_detector.load_model("models/dae_attack_detector_final")
    
    # 进行预测
    predictions, recon_errors, confidence = dae_detector.predict(X_test)
    
    # 评估
    metrics = dae_detector._compute_evaluation_metrics(y_test, predictions, recon_errors)
    
    # 可视化
    dae_detector._visualize_evaluation_results(y_test, predictions, recon_errors)
    
    return dae_detector


if __name__ == "__main__":
    # 运行完整的DAE攻击检测流程
    detector = run_dae_attack_detection()
    
    # 或者测试预训练模型
    # detector = test_pretrained_model()