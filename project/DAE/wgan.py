# wgan_generator.py
"""
WGAN生成器用于生成虚假数据注入攻击的异常数据
用于数据增强，解决样本稀缺问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Generator(nn.Module):
    """WGAN生成器网络"""
    
    def __init__(self, latent_dim=100, output_dim=84, hidden_dims=[256, 512, 256]):
        """
        初始化生成器
        
        参数:
            latent_dim: 潜在空间维度
            output_dim: 输出维度（特征数）
            hidden_dims: 隐藏层维度列表
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 构建网络层
        layers = []
        input_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())  # 使用Tanh将输出限制在[-1,1]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z):
        """前向传播"""
        return self.model(z)
    
    def generate(self, num_samples, device='cpu'):
        """生成样本"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self(z)
        return samples.cpu().numpy()


class Critic(nn.Module):
    """WGAN判别器（批评家）"""
    
    def __init__(self, input_dim=84, hidden_dims=[256, 128, 64]):
        """
        初始化判别器
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
        """
        super(Critic, self).__init__()
        
        # 构建网络层（WGAN不使用批归一化在判别器中）
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim = hidden_dim
        
        # 输出层：WGAN输出实数，不是概率
        layers.append(nn.Linear(current_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)


class Classifier(nn.Module):
    """辅助分类器，用于指导生成器生成特定类别的数据"""
    
    def __init__(self, input_dim=84, hidden_dims=[128, 64], num_classes=2):
        """
        初始化分类器
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            num_classes: 类别数
        """
        super(Classifier, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)


class WGAN_GP:
    """WGAN with Gradient Penalty (WGAN-GP)"""
    
    def __init__(self, generator, critic, classifier=None, latent_dim=100, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化WGAN-GP
        
        参数:
            generator: 生成器
            critic: 判别器
            classifier: 分类器（可选）
            latent_dim: 潜在空间维度
            device: 计算设备
        """
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.classifier = classifier.to(device) if classifier else None
        
        self.latent_dim = latent_dim
        self.device = device
        
        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
        if self.classifier:
            self.clf_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
            self.clf_criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.g_losses = []
        self.c_losses = []
        self.gp_losses = []
        
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """
        计算梯度惩罚（WGAN-GP的关键）
        
        参数:
            real_samples: 真实样本
            fake_samples: 生成样本
        """
        batch_size = real_samples.size(0)
        
        # 生成随机权重
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        
        # 插值样本
        interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # 计算判别器对插值样本的输出
        d_interpolated = self.critic(interpolated)
        
        # 计算梯度
        grad_outputs = torch.ones_like(d_interpolated, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_data, target_labels=None, lambda_gp=10, lambda_cls=1.0):
        """
        训练一步
        
        参数:
            real_data: 真实数据
            target_labels: 目标标签（用于分类器指导）
            lambda_gp: 梯度惩罚系数
            lambda_cls: 分类器损失系数
        """
        batch_size = real_data.size(0)
        
        # ========== 训练判别器 ==========
        self.critic.train()
        self.generator.eval()
        
        # 生成假数据
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z).detach()
        
        # 计算判别器损失
        real_output = self.critic(real_data)
        fake_output = self.critic(fake_data)
        
        # Wasserstein距离
        wasserstein_distance = real_output.mean() - fake_output.mean()
        
        # 梯度惩罚
        gradient_penalty = self.compute_gradient_penalty(real_data, fake_data)
        
        # 总损失
        c_loss = -wasserstein_distance + lambda_gp * gradient_penalty
        
        # 反向传播
        self.c_optimizer.zero_grad()
        c_loss.backward()
        self.c_optimizer.step()
        
        # ========== 训练生成器 ==========
        if self.classifier and target_labels is not None:
            # 每5步训练一次生成器（WGAN标准做法）
            if len(self.g_losses) % 5 == 0:
                self.critic.eval()
                self.generator.train()
                
                # 生成新数据
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                generated_data = self.generator(z)
                
                # 判别器损失
                g_loss_d = -self.critic(generated_data).mean()
                
                # 分类器损失：确保生成的数据被分类为目标类别（攻击数据）
                if self.classifier:
                    classifier_output = self.classifier(generated_data)
                    g_loss_cls = self.clf_criterion(classifier_output, target_labels)
                    g_loss = g_loss_d + lambda_cls * g_loss_cls
                else:
                    g_loss = g_loss_d
                
                # 反向传播
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                
                # 记录损失
                self.g_losses.append(g_loss.item())
        
        # 记录损失
        self.c_losses.append(c_loss.item())
        self.gp_losses.append(gradient_penalty.item())
        
        return {
            'c_loss': c_loss.item(),
            'w_distance': wasserstein_distance.item(),
            'gp_loss': gradient_penalty.item(),
            'g_loss': self.g_losses[-1] if len(self.g_losses) > 0 else 0
        }
    
    def train_classifier(self, train_loader, epochs=10):
        """预训练分类器"""
        if not self.classifier:
            print("没有分类器可训练")
            return
        
        self.classifier.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for data, labels in train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device).long()
                
                # 前向传播
                outputs = self.classifier(data)
                loss = self.clf_criterion(outputs, labels)
                
                # 反向传播
                self.clf_optimizer.zero_grad()
                loss.backward()
                self.clf_optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            print(f'分类器训练 Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
    
    def generate_attack_data(self, num_samples, target_class=1):
        """
        生成攻击数据
        
        参数:
            num_samples: 生成样本数量
            target_class: 目标类别（1=攻击，0=正常）
        """
        self.generator.eval()
        
        # 生成潜在向量
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        # 生成数据
        with torch.no_grad():
            generated_data = self.generator(z)
        
        # 如果需要，使用分类器进行筛选
        if self.classifier:
            with torch.no_grad():
                predictions = self.classifier(generated_data)
                _, predicted_classes = torch.max(predictions, 1)
            
            # 筛选出被分类为目标类别的样本
            target_indices = (predicted_classes == target_class).nonzero().squeeze()
            
            if target_indices.numel() > 0:
                if target_indices.dim() == 0:
                    target_indices = target_indices.unsqueeze(0)
                generated_data = generated_data[target_indices]
        
        return generated_data.cpu().numpy()
    
    def save_models(self, path="models/wgan"):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.generator.state_dict(), f"{path}/generator.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        
        if self.classifier:
            torch.save(self.classifier.state_dict(), f"{path}/classifier.pth")
        
        # 保存训练历史
        history = {
            'g_losses': self.g_losses,
            'c_losses': self.c_losses,
            'gp_losses': self.gp_losses
        }
        
        with open(f"{path}/training_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        print(f"模型保存到 {path}")
    
    def load_models(self, path="models/wgan"):
        """加载模型"""
        self.generator.load_state_dict(torch.load(f"{path}/generator.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth", map_location=self.device))
        
        if self.classifier and os.path.exists(f"{path}/classifier.pth"):
            self.classifier.load_state_dict(torch.load(f"{path}/classifier.pth", map_location=self.device))
        
        print(f"模型从 {path} 加载")
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 判别器损失
        axes[0, 0].plot(self.c_losses, label='Critic Loss', alpha=0.7)
        axes[0, 0].set_title('判别器损失')
        axes[0, 0].set_xlabel('迭代')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 生成器损失
        if len(self.g_losses) > 0:
            axes[0, 1].plot(self.g_losses, label='Generator Loss', color='orange', alpha=0.7)
            axes[0, 1].set_title('生成器损失')
            axes[0, 1].set_xlabel('迭代')
            axes[0, 1].set_ylabel('损失')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 梯度惩罚
        axes[1, 0].plot(self.gp_losses, label='Gradient Penalty', color='green', alpha=0.7)
        axes[1, 0].set_title('梯度惩罚')
        axes[1, 0].set_xlabel('迭代')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 损失对比
        axes[1, 1].plot(self.c_losses[:min(1000, len(self.c_losses))], label='Critic', alpha=0.7)
        if len(self.g_losses) > 0:
            axes[1, 1].plot(self.g_losses[:min(1000, len(self.g_losses))], label='Generator', alpha=0.7)
        axes[1, 1].set_title('损失对比')
        axes[1, 1].set_xlabel('迭代')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"训练历史图保存到 {save_path}")
        
        plt.show()


class AttackDataGenerator:
    """攻击数据生成器，使用WGAN生成虚假数据"""
    
    def __init__(self, feature_dim=84, latent_dim=100, device=None):
        """
        初始化攻击数据生成器
        
        参数:
            feature_dim: 特征维度
            latent_dim: 潜在空间维度
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 初始化模型
        self.generator = Generator(latent_dim=latent_dim, output_dim=feature_dim)
        self.critic = Critic(input_dim=feature_dim)
        self.classifier = Classifier(input_dim=feature_dim)
        
        # 初始化WGAN
        self.wgan = WGAN_GP(
            generator=self.generator,
            critic=self.critic,
            classifier=self.classifier,
            latent_dim=latent_dim,
            device=self.device
        )
        
        # 数据存储
        self.generated_data = None
        self.generated_labels = None
    
    def prepare_attack_data(self, X_train, y_train):
        """
        准备攻击数据用于训练
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
        """
        # 提取攻击样本
        attack_indices = np.where(y_train == 1)[0]
        attack_data = X_train[attack_indices]
        
        print(f"攻击样本数量: {len(attack_data)}")
        print(f"攻击样本形状: {attack_data.shape}")
        
        # 如果数据是窗口数据，取最后一个时间步（或平均值）
        if len(attack_data.shape) == 3:  # (batch, window, features)
            print("检测到窗口数据，转换为单个时间步...")
            # 取窗口最后一个时间步
            attack_data = attack_data[:, -1, :]
            print(f"转换后形状: {attack_data.shape}")
        
        # 转换为PyTorch张量
        attack_tensor = torch.FloatTensor(attack_data).to(self.device)
        
        # 创建数据加载器
        attack_dataset = TensorDataset(attack_tensor, torch.ones(len(attack_tensor), dtype=torch.long))
        attack_loader = DataLoader(attack_dataset, batch_size=32, shuffle=True)
        
        return attack_loader
    
    def train(self, X_train, y_train, epochs=1000, lambda_cls=1.0):
        """
        训练WGAN生成器
        
        参数:
            X_train: 训练数据
            y_train: 训练标签
            epochs: 训练轮数
            lambda_cls: 分类器损失系数
        """
        print("开始训练WGAN生成攻击数据...")
        print(f"设备: {self.device}")
        
        # 准备数据
        attack_loader = self.prepare_attack_data(X_train, y_train)
        
        # 预训练分类器（如果有正常和攻击数据）
        print("\n预训练分类器...")
        normal_indices = np.where(y_train == 0)[0]
        normal_data = X_train[normal_indices]
        
        if len(normal_data.shape) == 3:
            normal_data = normal_data[:, -1, :]
        
        # 合并数据
        all_data = np.vstack([normal_data, attack_loader.dataset.tensors[0].cpu().numpy()])
        all_labels = np.concatenate([
            np.zeros(len(normal_data)),
            np.ones(len(attack_loader.dataset))
        ])
        
        # 创建分类器数据加载器
        clf_dataset = TensorDataset(
            torch.FloatTensor(all_data).to(self.device),
            torch.LongTensor(all_labels).to(self.device)
        )
        clf_loader = DataLoader(clf_dataset, batch_size=32, shuffle=True)
        
        # 训练分类器
        self.wgan.train_classifier(clf_loader, epochs=10)
        
        # 训练WGAN
        print("\n开始训练WGAN...")
        for epoch in range(epochs):
            total_c_loss = 0
            total_gp_loss = 0
            
            for batch_data, _ in attack_loader:
                batch_data = batch_data.to(self.device)
                
                # 目标标签（攻击数据，类别1）
                target_labels = torch.ones(batch_data.size(0), dtype=torch.long, device=self.device)
                
                # 训练一步
                losses = self.wgan.train_step(batch_data, target_labels, lambda_cls=lambda_cls)
                
                total_c_loss += losses['c_loss']
                total_gp_loss += losses['gp_loss']
            
            # 打印进度
            if (epoch + 1) % 100 == 0:
                avg_c_loss = total_c_loss / len(attack_loader)
                avg_gp_loss = total_gp_loss / len(attack_loader)
                
                print(f'Epoch [{epoch+1}/{epochs}], Critic Loss: {avg_c_loss:.4f}, GP Loss: {avg_gp_loss:.4f}')
        
        print("WGAN训练完成!")
        
        # 绘制训练历史
        self.wgan.plot_training_history(save_path="figures/wgan_training_history.png")
        
        # 保存模型
        self.wgan.save_models("models/wgan_attack_generator")
    
    def generate_data(self, num_samples=1000, target_class=1):
        """
        生成攻击数据
        
        参数:
            num_samples: 生成样本数量
            target_class: 目标类别
            
        返回:
            generated_data: 生成的数据
            generated_labels: 生成的标签
        """
        print(f"生成 {num_samples} 个攻击样本...")
        
        # 生成数据
        generated_data = self.wgan.generate_attack_data(num_samples, target_class)
        
        # 创建标签
        generated_labels = np.ones(len(generated_data), dtype=np.int32)
        
        print(f"生成完成: {generated_data.shape}")
        print(f"攻击样本数量: {len(generated_data)}")
        
        self.generated_data = generated_data
        self.generated_labels = generated_labels
        
        return generated_data, generated_labels
    
    def evaluate_generated_data(self, original_attack_data, save_path=None):
        """
        评估生成数据的质量
        
        参数:
            original_attack_data: 原始攻击数据
            save_path: 保存路径
        """
        if self.generated_data is None:
            print("请先生成数据")
            return
        
        # 确保维度一致
        if len(original_attack_data.shape) == 3:
            original_attack_data = original_attack_data[:, -1, :]
        
        # 选取部分样本进行可视化
        n_samples = min(5, len(original_attack_data), len(self.generated_data))
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        # 特征维度（假设前56维是原始特征，后28维是扩展特征）
        n_original_features = min(56, self.feature_dim)
        
        for i in range(n_samples):
            # 原始攻击数据
            axes[i, 0].bar(range(n_original_features), original_attack_data[i, :n_original_features], 
                          alpha=0.7, label='原始')
            axes[i, 0].set_title(f'样本 {i}: 原始攻击数据')
            axes[i, 0].set_xlabel('特征索引')
            axes[i, 0].set_ylabel('值')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # 生成攻击数据
            axes[i, 1].bar(range(n_original_features), self.generated_data[i, :n_original_features], 
                          alpha=0.7, color='orange', label='生成')
            axes[i, 1].set_title(f'样本 {i}: 生成攻击数据')
            axes[i, 1].set_xlabel('特征索引')
            axes[i, 1].set_ylabel('值')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"生成数据评估图保存到 {save_path}")
        
        plt.show()
        
        # 统计比较
        print("\n生成数据评估:")
        print(f"原始攻击数据均值: {np.mean(original_attack_data[:, :n_original_features], axis=0)[:5]}")
        print(f"生成攻击数据均值: {np.mean(self.generated_data[:, :n_original_features], axis=0)[:5]}")
        print(f"原始攻击数据标准差: {np.std(original_attack_data[:, :n_original_features], axis=0)[:5]}")
        print(f"生成攻击数据标准差: {np.std(self.generated_data[:, :n_original_features], axis=0)[:5]}")
        
        # 计算分布距离（JS散度）
        from scipy.spatial.distance import jensenshannon
        
        # 随机选择一些特征进行比较
        sample_features = np.random.choice(n_original_features, 5, replace=False)
        
        for feat_idx in sample_features:
            orig_dist = np.histogram(original_attack_data[:, feat_idx], bins=20, density=True)[0]
            gen_dist = np.histogram(self.generated_data[:, feat_idx], bins=20, density=True)[0]
            
            js_distance = jensenshannon(orig_dist, gen_dist)
            print(f"特征 {feat_idx} JS散度: {js_distance:.4f}")
    
    def save_generated_data(self, path="data/generated_attack_data.pkl"):
        """保存生成的数据"""
        if self.generated_data is None:
            print("没有生成的数据可保存")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data_dict = {
            'data': self.generated_data,
            'labels': self.generated_labels,
            'feature_dim': self.feature_dim,
            'description': 'WGAN生成的攻击数据'
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"生成数据保存到 {path}")
    
    def load_generated_data(self, path="data/generated_attack_data.pkl"):
        """加载生成的数据"""
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.generated_data = data_dict['data']
        self.generated_labels = data_dict['labels']
        
        print(f"生成数据从 {path} 加载")
        print(f"数据形状: {self.generated_data.shape}")
        
        return self.generated_data, self.generated_labels


def main():
    """主函数：演示WGAN生成攻击数据"""
    
    # 1. 加载之前处理的数据
    print("加载处理后的数据...")
    X_train = np.load("processed_data/X_train.npy")
    y_train = np.load("processed_data/y_train.npy")
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"训练标签形状: {y_train.shape}")
    
    # 2. 创建攻击数据生成器
    feature_dim = X_train.shape[2]  # 获取特征维度
    print(f"特征维度: {feature_dim}")
    
    generator = AttackDataGenerator(feature_dim=feature_dim, latent_dim=100)
    
    # 3. 训练WGAN（如果还没训练）
    generator.train(X_train, y_train, epochs=500, lambda_cls=0.5)
    
    # 或者加载已训练的模型
    # generator.wgan.load_models("models/wgan_attack_generator")
    
    # 4. 生成攻击数据
    generated_data, generated_labels = generator.generate_data(num_samples=2000)
    
    # 5. 评估生成数据
    # 提取原始攻击数据
    attack_indices = np.where(y_train == 1)[0]
    original_attack_data = X_train[attack_indices]
    
    generator.evaluate_generated_data(
        original_attack_data,
        save_path="figures/generated_data_evaluation.png"
    )
    
    # 6. 保存生成数据
    generator.save_generated_data()
    
    # 7. 将生成的数据集成到数据管道中
    # integrate_generated_data_to_pipeline(generated_data, generated_labels)


def integrate_generated_data_to_pipeline(generated_data, generated_labels, 
                                        original_data_path="processed_data"):
    """
    将生成的数据集成到数据管道中
    
    参数:
        generated_data: 生成的数据
        generated_labels: 生成的标签
        original_data_path: 原始数据路径
    """
    print("\n将生成数据集成到数据管道...")
    
    # 加载原始数据
    X_train = np.load(f"{original_data_path}/X_train.npy")
    y_train = np.load(f"{original_data_path}/y_train.npy")
    
    # 转换生成数据为窗口格式
    # 假设生成的数据是单个时间步，我们需要将其转换为窗口
    window_size = X_train.shape[1]
    feature_dim = X_train.shape[2]
    
    # 创建滑动窗口（简单重复）
    n_samples = len(generated_data)
    generated_windows = np.zeros((n_samples, window_size, feature_dim))
    
    for i in range(n_samples):
        # 用生成的数据填充整个窗口
        generated_windows[i] = np.tile(generated_data[i], (window_size, 1))
    
    print(f"生成的窗口数据形状: {generated_windows.shape}")
    
    # 合并数据
    X_train_augmented = np.concatenate([X_train, generated_windows], axis=0)
    y_train_augmented = np.concatenate([y_train, generated_labels], axis=0)
    
    print(f"增强后训练数据形状: {X_train_augmented.shape}")
    print(f"增强后训练标签形状: {y_train_augmented.shape}")
    
    # 统计类别分布
    n_normal = np.sum(y_train_augmented == 0)
    n_attack = np.sum(y_train_augmented == 1)
    
    print(f"\n增强后类别分布:")
    print(f"正常样本: {n_normal} ({n_normal/len(y_train_augmented)*100:.2f}%)")
    print(f"攻击样本: {n_attack} ({n_attack/len(y_train_augmented)*100:.2f}%)")
    print(f"不平衡比例: {n_normal/max(n_attack, 1):.2f}:1")
    
    # 保存增强后的数据
    save_dir = "processed_data_augmented"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f"{save_dir}/X_train.npy", X_train_augmented)
    np.save(f"{save_dir}/y_train.npy", y_train_augmented)
    
    # 复制其他数据
    for file_name in ['X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']:
        src_path = f"{original_data_path}/{file_name}"
        dst_path = f"{save_dir}/{file_name}"
        
        if os.path.exists(src_path):
            import shutil
            shutil.copy(src_path, dst_path)
    
    print(f"\n增强数据保存到 {save_dir}")
    print("数据增强完成！")


if __name__ == "__main__":
    # 测试WGAN
    main()
    
    # 或者直接集成已生成的数据
    # 首先加载生成的数据
    # with open("data/attack_single_point.pkl", 'rb') as f:
    #     data_dict = pickle.load(f)
    
    # generated_data = data_dict['data']
    # generated_labels = data_dict['labels']
    
    # # 集成到数据管道
    # integrate_generated_data_to_pipeline(generated_data, generated_labels)