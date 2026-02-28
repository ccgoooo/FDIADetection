import numpy as np
import pandas as pd
import json
import os

class FDIA_DataGenerator:
    """FDIA数据生成器"""
    
    def __init__(self, random_seed=42):
        """初始化生成器"""
        np.random.seed(random_seed)
        
    def generate_normal_data(self, num_samples=10000, num_features=14):
        """生成纯正常数据"""
        print(f"正在生成 {num_samples} 条纯正常数据...")
        
        data = []
        for i in range(num_samples):
            # 生成正常电压数据（符合电力系统稳态特性）
            # 大部分电压在1.0附近，少量波动
            base_voltage = 1.0
            
            # 添加日内负荷变化效应（模拟一天不同时段）
            time_of_day = i % 24
            daily_variation = 0.02 * np.sin(2 * np.pi * time_of_day / 24)
            
            # 添加随机波动
            random_variation = np.random.normal(0, 0.015, num_features)
            
            # 生成最终电压值
            voltages = base_voltage + daily_variation + random_variation
            
            # 确保电压在合理范围内（0.95-1.05 pu）
            voltages = np.clip(voltages, 0.95, 1.05)
            
            data.append(voltages)
        
        # 转换为DataFrame
        df = pd.DataFrame(data, columns=[f'Bus_{i+1}' for i in range(num_features)])
        df['label'] = 0  # 全部标记为正常
        
        print(f"✅ 纯正常数据生成完成！")
        print(f"   样本数: {len(df)}")
        print(f"   特征数: {num_features}")
        print(f"   电压范围: [{df.values[:, :num_features].min():.3f}, {df.values[:, :num_features].max():.3f}]")
        
        return df
    
    def inject_attack(self, voltages, attack_type, timestamp=0):
        """注入特定类型的攻击"""
        num_features = len(voltages)
        attacked_voltages = voltages.copy()
        
        if attack_type == "random":
            # 随机攻击：随机篡改1-3个节点的电压
            num_targets = np.random.randint(1, 4)
            targets = np.random.choice(num_features, num_targets, replace=False)
            for target in targets:
                # 较大的随机篡改（±20%）
                attack_factor = np.random.uniform(0.8, 1.2)
                attacked_voltages[target] *= attack_factor
        
        elif attack_type == "stealthy":
            # 隐蔽攻击：单个节点缓慢漂移
            target = np.random.randint(0, num_features)
            # 小幅度缓慢变化（最大±5%）
            drift = 1.0 + 0.05 * np.sin(timestamp / 5.0 + np.random.random())
            attacked_voltages[target] *= drift
        
        elif attack_type == "coordinated":
            # 协同攻击：两个相关节点协同变化
            # 选择两个相邻节点（假设Bus_1和Bus_2相关）
            if num_features >= 2:
                attacked_voltages[0] *= 0.85  # 大幅降低
                attacked_voltages[1] *= 1.15  # 大幅升高
                # 保持总功率平衡的趋势
        
        return attacked_voltages
    
    def generate_data_with_attacks(self, num_samples=10000, num_features=14, attack_ratio=0.2):
        """生成包含攻击的数据"""
        print(f"正在生成 {num_samples} 条数据，攻击比例 {attack_ratio*100}%...")
        
        data = []
        labels = []
        attack_details = []
        
        # 攻击类型分布
        attack_types = ["random", "stealthy", "coordinated"]
        attack_type_distribution = [0.4, 0.3, 0.3]  # 40%随机，30%隐蔽，30%协同
        
        normal_count = 0
        attack_count = 0
        
        for i in range(num_samples):
            # 生成正常电压数据（与纯正常数据相同）
            base_voltage = 1.0
            time_of_day = i % 24
            daily_variation = 0.02 * np.sin(2 * np.pi * time_of_day / 24)
            random_variation = np.random.normal(0, 0.015, num_features)
            
            voltages = base_voltage + daily_variation + random_variation
            voltages = np.clip(voltages, 0.95, 1.05)
            
            # 决定是否注入攻击
            is_attack = np.random.random() < attack_ratio
            
            if is_attack:
                # 选择攻击类型
                attack_type = np.random.choice(attack_types, p=attack_type_distribution)
                
                # 注入攻击
                attacked_voltages = self.inject_attack(voltages, attack_type, i)
                
                # 确保攻击后的电压仍在合理范围内（但可能有明显异常）
                attacked_voltages = np.clip(attacked_voltages, 0.85, 1.15)
                
                data.append(attacked_voltages)
                labels.append(1)
                attack_count += 1
                
                # 记录攻击详情
                detail = {
                    'sample_id': i,
                    'attack_type': attack_type,
                    'original_voltage_mean': np.mean(voltages),
                    'attacked_voltage_mean': np.mean(attacked_voltages),
                    'max_change': np.max(np.abs(attacked_voltages - voltages))
                }
                attack_details.append(detail)
            else:
                data.append(voltages)
                labels.append(0)
                normal_count += 1
                
                attack_details.append({
                    'sample_id': i,
                    'attack_type': 'normal'
                })
        
        # 转换为DataFrame
        df = pd.DataFrame(data, columns=[f'Bus_{i+1}' for i in range(num_features)])
        df['label'] = labels
        
        # 计算攻击统计
        attack_by_type = {}
        for detail in attack_details:
            if detail['attack_type'] != 'normal':
                attack_by_type[detail['attack_type']] = attack_by_type.get(detail['attack_type'], 0) + 1
        
        print(f"✅ 包含攻击的数据生成完成！")
        print(f"   总样本数: {len(df)}")
        print(f"   正常样本: {normal_count} ({normal_count/num_samples*100:.1f}%)")
        print(f"   攻击样本: {attack_count} ({attack_count/num_samples*100:.1f}%)")
        
        if attack_by_type:
            print("   攻击类型分布:")
            for attack_type, count in attack_by_type.items():
                percentage = count/attack_count*100
                print(f"     - {attack_type}: {count} ({percentage:.1f}%)")
        
        print(f"   电压范围: [{df.values[:, :num_features].min():.3f}, {df.values[:, :num_features].max():.3f}]")
        
        return df, attack_details
    
    def save_dataset(self, df, filename, metadata=None):
        """保存数据集到CSV文件"""
        # 确保data目录存在
        os.makedirs('data', exist_ok=True)
        
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False)
        
        print(f"💾 数据集已保存到: {filepath}")
        print(f"   文件大小: {os.path.getsize(filepath)/1024/1024:.2f} MB")
        
        # 保存元数据（如果有）
        if metadata:
            metadata_path = f"data/{filename.replace('.csv', '_metadata.json')}"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   元数据: {metadata_path}")
        
        return filepath
    
    def analyze_dataset(self, df):
        """分析数据集特征"""
        num_features = len(df.columns) - 1  # 减去标签列
        
        print("\n📈 数据集分析:")
        print("=" * 50)
        
        # 标签分布
        label_counts = df['label'].value_counts()
        print("标签分布:")
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            label_name = "攻击" if label == 1 else "正常"
            print(f"  {label_name} (标签={label}): {count} 条 ({percentage:.1f}%)")
        
        # 特征统计
        print("\n特征统计 (前5个节点):")
        for i in range(min(5, num_features)):
            col_name = f'Bus_{i+1}'
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            min_val = df[col_name].min()
            max_val = df[col_name].max()
            print(f"  {col_name}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 范围=[{min_val:.3f}, {max_val:.3f}]")
        
        # 攻击检测难度分析（仅当有攻击数据时）
        if 1 in df['label'].values:
            normal_data = df[df['label'] == 0].iloc[:, :num_features]
            attack_data = df[df['label'] == 1].iloc[:, :num_features]
            
            normal_mean = normal_data.mean(axis=1).mean()
            attack_mean = attack_data.mean(axis=1).mean()
            
            print(f"\n攻击检测难度分析:")
            print(f"  正常数据平均电压: {normal_mean:.4f}")
            print(f"  攻击数据平均电压: {attack_mean:.4f}")
            print(f"  差异: {abs(attack_mean - normal_mean):.4f}")
            
            # 计算可分离性指标
            from scipy.spatial.distance import mahalanobis
            try:
                # 使用马氏距离估计可分离性
                cov_matrix = np.cov(df.iloc[:, :num_features].T)
                cov_inv = np.linalg.pinv(cov_matrix)
                
                normal_center = normal_data.mean().values
                attack_center = attack_data.mean().values
                
                distance = mahalanobis(normal_center, attack_center, cov_inv)
                print(f"  马氏距离（可分离性）: {distance:.2f}")
                
                if distance > 3:
                    print("  结论: 两类数据相对容易分离")
                elif distance > 1:
                    print("  结论: 两类数据有一定可分离性")
                else:
                    print("  结论: 两类数据较难分离（攻击隐蔽性高）")
            except:
                print("  注意: 无法计算精确的可分离性指标")

# 主程序
if __name__ == "__main__":
    # 创建生成器
    generator = FDIA_DataGenerator(random_seed=42)
    
    print("=" * 60)
    print("FDIA数据集生成器")
    print("=" * 60)
    
    # 生成纯正常数据
    print("\n1. 生成纯正常数据集...")
    normal_df = generator.generate_normal_data(num_samples=10000, num_features=14)
    normal_file = generator.save_dataset(normal_df, "fdia_data_normal_10000.csv")
    
    # 分析纯正常数据集
    generator.analyze_dataset(normal_df)
    
    # 生成包含20%攻击的数据
    print("\n\n2. 生成包含20%攻击的数据集...")
    attack_df, attack_details = generator.generate_data_with_attacks(
        num_samples=10000, 
        num_features=14, 
        attack_ratio=0.2
    )
    
    # 保存攻击数据集
    attack_metadata = {
        'total_samples': 10000,
        'normal_samples': int(np.sum(attack_df['label'] == 0)),
        'attack_samples': int(np.sum(attack_df['label'] == 1)),
        'attack_ratio': 0.2,
        'feature_dim': 14,
        'attack_type_distribution': {
            'random': 0.4,
            'stealthy': 0.3,
            'coordinated': 0.3
        }
    }
    
    attack_file = generator.save_dataset(
        attack_df, 
        "fdia_data_attack_20_10000.csv",
        metadata=attack_metadata
    )
    
    # 分析攻击数据集
    generator.analyze_dataset(attack_df)
    
    print("\n" + "=" * 60)
    print("数据集生成完成！")
    print("=" * 60)
    print(f"1. 纯正常数据: {normal_file}")
    print(f"   用途: 模型训练、基线测试")
    print(f"   特点: 100%正常数据，无攻击")
    
    print(f"\n2. 包含攻击数据: {attack_file}")
    print(f"   用途: 攻击检测模型训练和测试")
    print(f"   特点: 80%正常 + 20%攻击（随机40%、隐蔽30%、协同30%）")
    
    print("\n🎯 建议使用方式:")
    print("   1. 用纯正常数据训练自编码器等无监督模型")
    print("   2. 用包含攻击数据训练有监督分类模型（CNN、LSTM等）")
    print("   3. 用包含攻击数据测试模型性能")
    
    print("\n📁 所有文件已保存到 'data' 目录")
    print("=" * 60)