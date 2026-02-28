import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑、黑体
plt.rcParams['axes.unicode_minus'] = False

def basic_statistical_validation(df, dataset_name=""):
    """
    基础统计验证：检查数据范围、分布和异常
    """
    print(f"\n🔍 对数据集 [{dataset_name}] 进行基础统计验证")
    print("="*50)
    
    # 1. 分离特征和标签
    feature_cols = [col for col in df.columns if col.startswith('Bus_')]
    voltage_data = df[feature_cols]
    
    if 'label' in df.columns:
        labels = df['label']
        normal_data = voltage_data[labels == 0]
        attack_data = voltage_data[labels == 1]
    else:
        normal_data = voltage_data
        attack_data = None
    
    # 2. 数值范围检查（电力系统电压合理范围：0.9-1.1 pu）
    print("1. 电压数值范围检查:")
    min_val, max_val = voltage_data.min().min(), voltage_data.max().max()
    print(f"   全局范围: [{min_val:.3f}, {max_val:.3f}] pu")
    
    # 设定合理阈值
    lower_bound, upper_bound = 0.90, 1.10
    out_of_bounds = ((voltage_data < lower_bound) | (voltage_data > upper_bound)).sum().sum()
    total_values = voltage_data.size
    print(f"   超出[{lower_bound}, {upper_bound}]范围的比例: {out_of_bounds/total_values*100:.2f}%")
    if out_of_bounds/total_values > 0.01:  # 超过1%数据异常
        print("   ⚠️ 警告：异常电压值比例较高")
    else:
        print("   ✅ 电压范围基本合理")
    
    # 3. 分布形态检查（正常数据应接近正态分布）
    print("\n2. 分布形态检查（正常数据）:")
    # 取第一个节点的电压进行分布检验
    sample_voltages = normal_data.iloc[:, 0].values
    k2, p_value = stats.normaltest(sample_voltages)  # 正态性检验
    print(f"   节点1电压正态性检验 p值: {p_value:.4f}")
    if p_value > 0.05:
        print("   ✅ 数据接近正态分布（符合电力负荷波动特征）")
    else:
        print("   ⚠️ 数据分布与正态分布有显著差异")
    
    # 4. 统计特征
    print("\n3. 统计特征:")
    print(f"   平均值: {normal_data.mean().mean():.4f} ± {normal_data.std().mean():.4f}")
    print(f"   偏度（Skewness）: {normal_data.skew().mean():.4f} (接近0为对称)")
    print(f"   峰度（Kurtosis）: {normal_data.kurtosis().mean():.4f} (接近3为正态)")
    
    # 5. 如果是攻击数据，检查攻击引入的扰动
    if attack_data is not None and len(attack_data) > 0:
        print("\n4. 攻击数据扰动分析:")
        # 计算攻击引起的平均变化
        avg_normal = normal_data.mean().mean()
        avg_attack = attack_data.mean().mean()
        perturbation = abs(avg_attack - avg_normal) / avg_normal * 100
        print(f"   攻击引起的平均电压变化: {perturbation:.2f}%")
        
        # 检查攻击是否足够隐蔽（变化不宜过大）
        if perturbation < 5:
            print("   ✅ 攻击扰动较小，符合隐蔽性FDIA特征")
        else:
            print("   ⚠️ 攻击扰动较大，可能不够隐蔽")

def visualize_data_quality(df, dataset_name=""):
    """
    通过可视化检查数据质量
    """
    feature_cols = [col for col in df.columns if col.startswith('Bus_')]
    voltage_data = df[feature_cols]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'数据质量可视化 - {dataset_name}', fontsize=14)
    
    # 1. 电压分布直方图
    axes[0, 0].hist(voltage_data.values.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(x=1.0, color='red', linestyle='--', label='额定电压 (1.0 pu)')
    axes[0, 0].set_xlabel('电压值 (pu)')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('电压值全局分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 多个节点电压曲线（前100个样本）
    for i in range(min(5, len(feature_cols))):
        axes[0, 1].plot(voltage_data.iloc[:100, i], label=f'Bus_{i+1}', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('样本序号')
    axes[0, 1].set_ylabel('电压 (pu)')
    axes[0, 1].set_title('电压时序变化（前100样本）')
    axes[0, 1].legend(loc='upper right', fontsize='small')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 节点间相关性热图（前10个节点）
    corr_matrix = voltage_data.iloc[:, :10].corr()
    im = axes[0, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 2].set_title('节点电压相关性热图')
    axes[0, 2].set_xticks(range(len(corr_matrix.columns)))
    axes[0, 2].set_yticks(range(len(corr_matrix.index)))
    axes[0, 2].set_xticklabels([f'B{i+1}' for i in range(len(corr_matrix.columns))], rotation=45)
    axes[0, 2].set_yticklabels([f'B{i+1}' for i in range(len(corr_matrix.index))])
    plt.colorbar(im, ax=axes[0, 2])
    
    # 4. 箱线图查看异常值
    bp = axes[1, 0].boxplot(voltage_data.iloc[:, :8].values, tick_labels=[f'B{i+1}' for i in range(8)])
    axes[1, 0].set_ylabel('电压 (pu)')
    axes[1, 0].set_title('电压箱线图（检查异常值）')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 如果包含标签，显示正常vs攻击数据对比
    if 'label' in df.columns:
        labels = df['label']
        normal_voltage = voltage_data[labels == 0].iloc[:, 0].values[:500]
        attack_voltage = voltage_data[labels == 1].iloc[:, 0].values[:500]
        
        axes[1, 1].plot(normal_voltage, alpha=0.6, label='正常数据', linewidth=0.8)
        axes[1, 1].plot(attack_voltage, alpha=0.6, label='攻击数据', linewidth=0.8, color='red')
        axes[1, 1].set_xlabel('样本序号')
        axes[1, 1].set_ylabel('Bus_1 电压 (pu)')
        axes[1, 1].set_title('正常vs攻击数据对比（节点1）')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. QQ图检验正态性（正常数据）
    if 'label' in df.columns:
        sample_data = voltage_data[labels == 0].iloc[:, 0].values
    else:
        sample_data = voltage_data.iloc[:, 0].values
    
    stats.probplot(sample_data[:500], dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('QQ图（检验正态性）')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def validate_power_system_physics(df, line_from=[0,1,2], line_to=[1,2,3]):
    """
    验证电力系统物理规律（简化版）
    假设: line_from和line_to表示线路连接的节点索引
    """
    print("\n⚡ 电力系统物理规律验证")
    print("="*50)
    
    feature_cols = [col for col in df.columns if col.startswith('Bus_')]
    voltages = df[feature_cols].values
    
    violations = 0
    total_checks = 0
    
    # 规律1：相邻节点电压差异不应过大（一般<0.1 pu）
    print("1. 相邻节点电压差检查:")
    for i in range(len(line_from)):
        from_bus, to_bus = line_from[i], line_to[i]
        voltage_diff = np.abs(voltages[:, from_bus] - voltages[:, to_bus])
        
        # 检查是否超过阈值（0.1 pu是经验值）
        threshold = 0.15
        violation_count = np.sum(voltage_diff > threshold)
        violation_ratio = violation_count / len(voltage_diff) * 100
        
        print(f"   线路 {from_bus+1}-{to_bus+1}: {violation_ratio:.1f}% 样本电压差 > {threshold} pu")
        
        if violation_ratio > 5:  # 超过5%样本违反
            print(f"   ⚠️ 线路 {from_bus+1}-{to_bus+1} 电压差异常比例较高")
            violations += 1
        total_checks += 1
    
    # 规律2：系统平均电压应接近1.0 pu（标幺值系统）
    print("\n2. 系统平均电压检查:")
    avg_voltage = np.mean(voltages)
    print(f"   系统平均电压: {avg_voltage:.4f} pu")
    if 0.98 < avg_voltage < 1.02:
        print("   ✅ 系统平均电压在合理范围")
    else:
        print("   ⚠️ 系统平均电压偏离正常范围")
        violations += 1
    total_checks += 1
    
    # 规律3：电压变化应相对平滑（相邻样本间变化不应突变）
    print("\n3. 电压变化平滑性检查:")
    voltage_changes = np.diff(voltages[:100, :5], axis=0)  # 前100样本，前5个节点
    max_change = np.max(np.abs(voltage_changes))
    avg_change = np.mean(np.abs(voltage_changes))
    
    print(f"   最大瞬时变化: {max_change:.4f} pu")
    print(f"   平均瞬时变化: {avg_change:.4f} pu")
    
    if max_change < 0.05 and avg_change < 0.01:
        print("   ✅ 电压变化平滑，符合稳态运行特征")
    elif max_change > 0.1:
        print("   ⚠️ 电压存在突变，可能不符合稳态特征")
        violations += 1
    total_checks += 1
    
    # 总体评价
    print(f"\n{'='*50}")
    print(f"物理规律验证结果: {total_checks - violations}/{total_checks} 项通过")
    if violations == 0:
        print("✅ 数据基本符合电力系统物理规律")
    elif violations <= 2:
        print("⚠️  数据存在部分异常，但基本可用")
    else:
        print("❌ 数据存在较多物理规律违反，建议检查生成逻辑")



# 使用示例
if __name__ == "__main__":
    # 加载你生成的数据集
    normal_df = pd.read_csv('./data/fdia_data_normal_10000.csv')
    attack_df = pd.read_csv('./data/fdia_data_attack_20_10000.csv')
    
    # basic_statistical_validation(normal_df, "纯正常数据")
    # basic_statistical_validation(attack_df, "包含攻击数据")

    # # 可视化
    # visualize_data_quality(normal_df, "纯正常数据")
    # visualize_data_quality(attack_df, "包含攻击数据")

    # attack_df = pd.read_csv('./data/fdia_data_attack_20_10000.csv')
    # # 找出超出范围的样本
    # feature_cols = [col for col in attack_df.columns if col.startswith('Bus_')]
    # out_of_bound_mask = (attack_df[feature_cols] < 0.9).any(axis=1) | (attack_df[feature_cols] > 1.1).any(axis=1)
    # out_of_bound_samples = attack_df[out_of_bound_mask]
    # print(f"超出范围的样本中，攻击标签的比例：{out_of_bound_samples['label'].mean():.2%}")

    # 物理规律验证
    # # 使用示例（假设IEEE 14节点系统前几条线路）
    ieee14_from_bus = [0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 6, 6, 9, 9, 10, 12]
    ieee14_to_bus   = [1, 4, 2, 4, 3, 4, 4, 5, 7, 9, 7, 8, 10, 13, 11, 13]
    validate_power_system_physics(normal_df, ieee14_from_bus, ieee14_to_bus)
    validate_power_system_physics(attack_df, ieee14_from_bus, ieee14_to_bus)