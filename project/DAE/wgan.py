# correct_wgan_trainer.py
"""
正确的WGAN训练流程，包含分类器指导
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

class CorrectWGAN:
    """正确的WGAN训练实现"""
    
    def __init__(self, generator, critic, classifier, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.classifier = classifier.to(device)  # 分类器用于指导，不更新
        
        self.device = device
        
        # 优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
        # 损失记录
        self.g_losses = []
        self.c_losses = []
        
    def train_critic(self, real_data, n_critic=5):
        """训练Critic n次"""
        batch_size = real_data.size(0)
        c_loss_total = 0
        
        for _ in range(n_critic):
            # 生成假数据
            z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
            fake_data = self.generator(z).detach()
            
            # 计算Critic损失
            real_scores = self.critic(real_data)
            fake_scores = self.critic(fake_data)
            
            # WGAN损失：最大化真实数据与生成数据的Wasserstein距离
            c_loss = -(torch.mean(real_scores) - torch.mean(fake_scores))
            
            # 梯度惩罚
            gradient_penalty = self._gradient_penalty(real_data, fake_data)
            c_loss += 10 * gradient_penalty
            
            # 更新Critic
            self.c_optimizer.zero_grad()
            c_loss.backward()
            self.c_optimizer.step()
            
            c_loss_total += c_loss.item()
        
        return c_loss_total / n_critic
    
    def train_generator(self, batch_size, lambda_cls=0.1):
        """训练Generator，包含分类器指导"""
        # 生成假数据
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_data = self.generator(z)
        
        # Critic损失（WGAN损失）
        fake_scores = self.critic(fake_data)
        g_loss_critic = -torch.mean(fake_scores)
        
        # 分类器损失：鼓励生成的数据被分类为攻击（类别1）
        if self.classifier is not None:
            with torch.no_grad():  # 分类器不更新
                predictions = self.classifier(fake_data)
                # 我们希望生成的数据被分类为攻击类别（假设攻击=1）
                target_class = torch.ones(batch_size, dtype=torch.long, device=self.device)
                g_loss_cls = nn.CrossEntropyLoss()(predictions, target_class)
        else:
            g_loss_cls = 0
        
        # 总损失
        g_loss = g_loss_critic + lambda_cls * g_loss_cls
        
        # 更新Generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), g_loss_critic.item(), g_loss_cls if isinstance(g_loss_cls, float) else g_loss_cls.item()
    
    def _gradient_penalty(self, real_data, fake_data):
        """计算梯度惩罚（WGAN-GP）"""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device).expand_as(real_data)
        
        # 插值样本
        interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)
        
        # Critic对插值样本的评分
        scores = self.critic(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # 梯度惩罚：梯度范数偏离1的惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train(self, data_loader, epochs=1000, n_critic=5, lambda_cls=0.1):
        """完整训练流程"""
        print(f"开始训练WGAN，设备: {self.device}")
        print(f"Critic每轮训练次数: {n_critic}, 分类器权重: {lambda_cls}")
        
        for epoch in range(epochs):
            epoch_c_loss = 0
            epoch_g_loss = 0
            
            for batch_idx, (real_data, _) in enumerate(data_loader):
                real_data = real_data.to(self.device)
                
                # 训练Critic n次
                c_loss = self.train_critic(real_data, n_critic)
                epoch_c_loss += c_loss
                
                # 训练Generator 1次
                g_loss, g_loss_critic, g_loss_cls = self.train_generator(
                    real_data.size(0), lambda_cls
                )
                epoch_g_loss += g_loss
            
            # 记录损失
            avg_c_loss = epoch_c_loss / len(data_loader)
            avg_g_loss = epoch_g_loss / len(data_loader)
            
            self.c_losses.append(avg_c_loss)
            self.g_losses.append(avg_g_loss)
            
            # 打印进度
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"C Loss: {avg_c_loss:.4f}, G Loss: {avg_g_loss:.4f}")
                
                # 生成一些样本检查进度
                self._visualize_progress(epoch + 1)
    
    def _visualize_progress(self, epoch, n_samples=5):
        """可视化训练进度"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.generator.latent_dim, device=self.device)
            samples = self.generator(z).cpu().numpy()
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i in range(n_samples):
            axes[i].plot(samples[i, :20], alpha=0.7)  # 只显示前20个特征
            axes[i].set_title(f"生成样本 {i+1} (Epoch {epoch})")
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"figures/wgan_samples_epoch_{epoch}.png", dpi=150)
        plt.close()

class DataQualityEvaluator:
    """生成数据质量评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_statistical_similarity(self, real_data, fake_data):
        """
        评估统计相似性
        
        参数:
            real_data: 真实攻击数据 (n_samples, n_features)
            fake_data: 生成攻击数据 (n_samples, n_features)
        """
        results = {}
        
        # 1. 均值和方差比较
        real_mean = np.mean(real_data, axis=0)
        fake_mean = np.mean(fake_data, axis=0)
        real_std = np.std(real_data, axis=0)
        fake_std = np.std(fake_data, axis=0)
        
        results['mean_correlation'] = np.corrcoef(real_mean, fake_mean)[0, 1]
        results['std_correlation'] = np.corrcoef(real_std, fake_std)[0, 1]
        
        # 2. Jensen-Shannon散度（用于分布比较）
        from scipy.spatial.distance import jensenshannon
        
        js_distances = []
        for i in range(min(real_data.shape[1], fake_data.shape[1])):
            # 计算每个特征的JS散度
            real_hist = np.histogram(real_data[:, i], bins=50, density=True)[0]
            fake_hist = np.histogram(fake_data[:, i], bins=50, density=True)[0]
            js_dist = jensenshannon(real_hist, fake_hist)
            js_distances.append(js_dist)
        
        results['js_distance_mean'] = np.mean(js_distances)
        results['js_distance_std'] = np.std(js_distances)
        
        # 3. Wasserstein距离（Earth Mover's Distance）
        wasserstein_distances = []
        for i in range(min(real_data.shape[1], 10)):  # 只计算前10个特征，减少计算量
            w_dist = wasserstein_distance(real_data[:, i], fake_data[:, i])
            wasserstein_distances.append(w_dist)
        
        results['wasserstein_mean'] = np.mean(wasserstein_distances)
        
        return results
    
    def evaluate_feature_correlation(self, real_data, fake_data):
        """评估特征相关性结构"""
        # 计算特征相关矩阵
        real_corr = np.corrcoef(real_data.T)
        fake_corr = np.corrcoef(fake_data.T)
        
        # 相关矩阵的差异
        corr_diff = np.abs(real_corr - fake_corr)
        
        results = {
            'correlation_matrix_diff_mean': np.mean(corr_diff),
            'correlation_matrix_diff_max': np.max(corr_diff),
            'real_corr_rank': np.linalg.matrix_rank(real_corr),
            'fake_corr_rank': np.linalg.matrix_rank(fake_corr)
        }
        
        return results
    
    def evaluate_diversity(self, fake_data):
        """评估生成数据的多样性"""
        # 1. 样本间平均距离
        from scipy.spatial.distance import pdist
        
        # 随机采样以减少计算量
        n_samples = min(500, len(fake_data))
        indices = np.random.choice(len(fake_data), n_samples, replace=False)
        sampled_data = fake_data[indices]
        
        # 计算成对距离
        distances = pdist(sampled_data, metric='euclidean')
        
        results = {
            'avg_pairwise_distance': np.mean(distances),
            'std_pairwise_distance': np.std(distances),
            'distance_variation': np.var(distances)
        }
        
        # 2. 最近邻距离比率
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=2).fit(sampled_data)
        distances, _ = nbrs.kneighbors(sampled_data)
        
        # 最近邻距离（排除自身）
        nn_distances = distances[:, 1]
        results['avg_nearest_neighbor_distance'] = np.mean(nn_distances)
        
        return results
    
    def evaluate_classifier_performance(self, fake_data, classifier, target_class=1):
        """
        评估分类器对生成数据的性能
        
        参数:
            classifier: 预训练的分类器
            target_class: 我们希望生成的数据被分类为什么类别
        """
        # 确保分类器在评估模式
        classifier.eval()
        
        with torch.no_grad():
            fake_tensor = torch.FloatTensor(fake_data).to(classifier.device)
            predictions = classifier(fake_tensor)
            
            # 获取预测类别
            if predictions.shape[1] > 1:  # 多分类
                _, predicted_classes = torch.max(predictions, 1)
            else:  # 二分类
                predicted_classes = (predictions > 0.5).long().squeeze()
        
        # 计算准确率
        correct = (predicted_classes.cpu().numpy() == target_class).sum()
        accuracy = correct / len(fake_data)
        
        # 计算置信度
        if predictions.shape[1] > 1:
            probabilities = torch.softmax(predictions, dim=1)
            target_probs = probabilities[:, target_class].cpu().numpy()
        else:
            target_probs = torch.sigmoid(predictions).cpu().numpy().flatten()
        
        results = {
            'classification_accuracy': accuracy,
            'avg_confidence': np.mean(target_probs),
            'confidence_std': np.std(target_probs)
        }
        
        return results
    
    def visualize_comparison(self, real_data, fake_data, save_path=None):
        """可视化真实数据与生成数据的比较"""
        # 使用t-SNE降维可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # 合并数据并添加标签
        combined_data = np.vstack([real_data, fake_data])
        labels = np.array([0]*len(real_data) + [1]*len(fake_data))
        
        # t-SNE降维
        print("正在进行t-SNE降维...")
        tsne_results = tsne.fit_transform(combined_data)
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. t-SNE散点图
        axes[0, 0].scatter(tsne_results[labels==0, 0], tsne_results[labels==0, 1], 
                          alpha=0.5, label='真实攻击数据', s=20)
        axes[0, 0].scatter(tsne_results[labels==1, 0], tsne_results[labels==1, 1], 
                          alpha=0.5, label='生成攻击数据', s=20)
        axes[0, 0].set_title('t-SNE可视化：真实 vs 生成数据')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 特征分布对比（随机选3个特征）
        n_features = real_data.shape[1]
        sample_features = np.random.choice(n_features, 3, replace=False)
        
        for idx, feat_idx in enumerate(sample_features):
            axes[0, 1].hist(real_data[:, feat_idx], bins=50, alpha=0.5, 
                           density=True, label=f'真实-特征{feat_idx}')
            axes[0, 1].hist(fake_data[:, feat_idx], bins=50, alpha=0.5, 
                           density=True, label=f'生成-特征{feat_idx}')
        axes[0, 1].set_title('特征分布对比')
        axes[0, 1].set_xlabel('特征值')
        axes[0, 1].set_ylabel('密度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 均值比较（前20个特征）
        axes[1, 0].plot(np.mean(real_data, axis=0)[:20], 'o-', label='真实数据均值', alpha=0.7)
        axes[1, 0].plot(np.mean(fake_data, axis=0)[:20], 's-', label='生成数据均值', alpha=0.7)
        axes[1, 0].set_title('前20个特征均值对比')
        axes[1, 0].set_xlabel('特征索引')
        axes[1, 0].set_ylabel('均值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 相关矩阵差异热力图（前10个特征）
        real_corr = np.corrcoef(real_data[:, :10].T)
        fake_corr = np.corrcoef(fake_data[:, :10].T)
        corr_diff = np.abs(real_corr - fake_corr)
        
        im = axes[1, 1].imshow(corr_diff, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('特征相关矩阵差异（前10个特征）')
        axes[1, 1].set_xlabel('特征索引')
        axes[1, 1].set_ylabel('特征索引')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果保存到 {save_path}")
        
        plt.show()
        
        return tsne_results
    

"""
完整的WGAN数据生成与评估流程
"""

def complete_wgan_pipeline():
    """完整的WGAN数据生成流程"""
    
    # 1. 加载数据
    print("1. 加载数据...")
    X_train = np.load("processed_data/X_train.npy")
    y_train = np.load("processed_data/y_train.npy")
    
    # 提取攻击数据（假设攻击标签为1）
    attack_indices = np.where(y_train == 1)[0]
    attack_data = X_train[attack_indices]
    
    # 如果是窗口数据，转换为单时间步
    if len(attack_data.shape) == 3:
        attack_data = attack_data[:, -1, :]  # 取最后一个时间步
    
    print(f"攻击数据形状: {attack_data.shape}")
    
    # 2. 预训练分类器（如果还没有）
    print("\n2. 预训练分类器...")
    # 这里假设你已经有一个预训练好的分类器
    # 如果没有，需要先训练一个
    
    # 3. 训练WGAN
    print("\n3. 训练WGAN...")
    
    # 创建模型
    latent_dim = 100
    feature_dim = attack_data.shape[1]
    
    generator = Generator(latent_dim=latent_dim, output_dim=feature_dim)
    critic = Critic(input_dim=feature_dim)
    classifier = None  # 假设已经有预训练的分类器
    
    # 创建数据加载器
    attack_tensor = torch.FloatTensor(attack_data)
    attack_dataset = torch.utils.data.TensorDataset(attack_tensor, torch.zeros(len(attack_tensor)))
    attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=32, shuffle=True)
    
    # 训练WGAN
    wgan = CorrectWGAN(generator, critic, classifier)
    wgan.train(attack_loader, epochs=1000, n_critic=5, lambda_cls=0.1)
    
    # 4. 生成新数据
    print("\n4. 生成新数据...")
    n_samples = len(attack_data) * 2  # 生成两倍于原始攻击数据
    wgan.generator.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=wgan.device)
        generated_data = wgan.generator(z).cpu().numpy()
    
    print(f"生成数据形状: {generated_data.shape}")
    
    # 5. 评估生成数据质量
    print("\n5. 评估生成数据质量...")
    evaluator = DataQualityEvaluator()
    
    # 统计相似性评估
    stats = evaluator.evaluate_statistical_similarity(attack_data, generated_data)
    print("\n统计相似性评估:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 特征相关性评估
    corr_stats = evaluator.evaluate_feature_correlation(attack_data, generated_data)
    print("\n特征相关性评估:")
    for key, value in corr_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 多样性评估
    diversity_stats = evaluator.evaluate_diversity(generated_data)
    print("\n多样性评估:")
    for key, value in diversity_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # 可视化比较
    print("\n生成可视化比较...")
    tsne_results = evaluator.visualize_comparison(
        attack_data[:500],  # 只取一部分用于可视化
        generated_data[:500],
        save_path="figures/real_vs_generated_comparison.png"
    )
    
    # 6. 保存生成数据
    print("\n6. 保存生成数据...")
    np.save("data/generated_attack_data.npy", generated_data)
    
    # 7. 生成质量报告
    generate_quality_report(attack_data, generated_data, stats, corr_stats, diversity_stats)
    
    return generated_data

def generate_quality_report(real_data, fake_data, stats, corr_stats, diversity_stats):
    """生成质量评估报告"""
    report = """
    =============================================
    WGAN生成数据质量评估报告
    =============================================
    
    1. 数据基本信息
       - 真实攻击数据: {} 样本, {} 特征
       - 生成攻击数据: {} 样本, {} 特征
    
    2. 统计相似性
       - 均值相关性: {:.4f}
       - 标准差相关性: {:.4f}
       - JS散度均值: {:.4f}
       - Wasserstein距离均值: {:.4f}
    
    3. 特征相关性
       - 相关矩阵差异均值: {:.4f}
       - 相关矩阵差异最大值: {:.4f}
       - 真实数据相关矩阵秩: {}
       - 生成数据相关矩阵秩: {}
    
    4. 多样性评估
       - 平均成对距离: {:.4f}
       - 最近邻平均距离: {:.4f}
       - 距离方差: {:.4f}
    
    5. 评估结论
    """.format(
        len(real_data), real_data.shape[1],
        len(fake_data), fake_data.shape[1],
        stats.get('mean_correlation', 0),
        stats.get('std_correlation', 0),
        stats.get('js_distance_mean', 0),
        stats.get('wasserstein_mean', 0),
        corr_stats.get('correlation_matrix_diff_mean', 0),
        corr_stats.get('correlation_matrix_diff_max', 0),
        corr_stats.get('real_corr_rank', 0),
        corr_stats.get('fake_corr_rank', 0),
        diversity_stats.get('avg_pairwise_distance', 0),
        diversity_stats.get('avg_nearest_neighbor_distance', 0),
        diversity_stats.get('distance_variation', 0)
    )
    
    # 添加结论
    if stats.get('mean_correlation', 0) > 0.8:
        report += "    - 统计特性保持良好（均值相关性>0.8）\n"
    else:
        report += "    - 警告：统计特性保持不佳\n"
    
    if stats.get('js_distance_mean', 0) < 0.2:
        report += "    - 分布相似性高（JS散度<0.2）\n"
    else:
        report += "    - 警告：分布相似性不足\n"
    
    if diversity_stats.get('avg_pairwise_distance', 0) > 0.1:
        report += "    - 生成数据多样性充足\n"
    else:
        report += "    - 警告：生成数据多样性不足，可能存在模式崩溃\n"
    
    # 保存报告
    with open("reports/wgan_quality_report.txt", "w") as f:
        f.write(report)
    
    print(report)
    print("\n详细报告已保存到: reports/wgan_quality_report.txt")