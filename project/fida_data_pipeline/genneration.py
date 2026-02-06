import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

class SimpleFDIA_DataGenerator:
    """简化的FDIA数据生成器"""
    
    def __init__(self):
        """初始化IEEE 14总线系统"""
        self.net = pn.case14()
        self._identify_bus_types()
        print(f"系统节点总数: {len(self.net.bus)}")
        print(f"平衡节点: {self.slack_bus}")
        print(f"零注入节点: {self.zero_injection_nodes}")
        print(f"可攻击节点: {self.attackable_nodes}")
    
    def _identify_bus_types(self):
        """识别节点类型"""
        # 平衡节点（通常第一个是slack）
        self.slack_bus = 0  # IEEE 14中节点1
        
        # 识别零注入节点（P=0且没有负荷）
        self.zero_injection_nodes = []
        for bus_idx in range(len(self.net.bus)):
            # 检查是否有负荷或发电机
            has_load = bus_idx in self.net.load.bus.values
            has_gen = bus_idx in self.net.gen.bus.values
            if not has_load and not has_gen and bus_idx != self.slack_bus:
                self.zero_injection_nodes.append(bus_idx)
        
        # 可攻击节点（排除平衡节点和零注入节点）
        self.attackable_nodes = [
            i for i in range(len(self.net.bus)) 
            if i not in [self.slack_bus] + self.zero_injection_nodes
        ]
    
    def run_ac_power_flow(self, load_variation=0.1, line_outage=None):
        """
        运行交流潮流计算
        
        参数:
        load_variation: 负荷波动幅度 (±百分比)
        line_outage: 断开的线路 [(from_bus, to_bus)]
        
        返回:
        测量数据字典
        """
        # 备份原始网络
        net_backup = self.net.deepcopy()
        
        try:
            # 1. 添加负荷波动
            if load_variation > 0:
                for idx in self.net.load.index:
                    variation = 1 + np.random.uniform(-load_variation, load_variation)
                    self.net.load.p_mw[idx] *= variation
                    self.net.load.q_mvar[idx] *= variation
            
            # 2. 模拟线路断开
            if line_outage:
                for from_bus, to_bus in line_outage:
                    # 找到对应线路
                    line_idx = self.net.line[
                        (self.net.line.from_bus == from_bus) & 
                        (self.net.line.to_bus == to_bus)
                    ]
                    if len(line_idx) > 0:
                        self.net.line.in_service[line_idx.index[0]] = False
            
            # 3. 运行AC潮流
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            
            # 4. 收集测量数据
            measurements = self._collect_measurements()
            
            # 恢复网络
            self.net = net_backup
            
            return measurements
            
        except Exception as e:
            print(f"AC潮流计算失败: {e}")
            self.net = net_backup
            return None
    
    def run_dc_power_flow(self, load_variation=0.1, line_outage=None):
        """
        运行直流潮流计算（近似）
        
        注意：pandapower没有专门的DC潮流，我们使用AC潮流但固定电压幅值
        """
        # 备份原始网络
        net_backup = self.net.deepcopy()
        
        try:
            # 1. 添加负荷波动
            if load_variation > 0:
                for idx in self.net.load.index:
                    variation = 1 + np.random.uniform(-load_variation, load_variation)
                    self.net.load.p_mw[idx] *= variation
                    # DC潮流通常忽略无功
                    self.net.load.q_mvar[idx] = 0
            
            # 2. 模拟线路断开
            if line_outage:
                for from_bus, to_bus in line_outage:
                    line_idx = self.net.line[
                        (self.net.line.from_bus == from_bus) & 
                        (self.net.line.to_bus == to_bus)
                    ]
                    if len(line_idx) > 0:
                        self.net.line.in_service[line_idx.index[0]] = False
            
            # 3. 运行DC近似潮流
            # 设置所有发电机为PV节点，电压固定
            for idx in self.net.gen.index:
                self.net.gen.vm_pu[idx] = 1.0
                self.net.gen.min_q_mvar[idx] = -999
                self.net.gen.max_q_mvar[idx] = 999
            
            # 运行潮流
            pp.rundcpp(self.net)
            
            # 4. 收集测量数据（只收集有功和相角）
            measurements = self._collect_dc_measurements()
            
            # 恢复网络
            self.net = net_backup
            
            return measurements
            
        except Exception as e:
            print(f"DC潮流计算失败: {e}")
            self.net = net_backup
            return None
    
    def _collect_measurements(self):
        """收集AC测量数据"""
        measurements = {}
        
        # 节点测量
        for bus_idx in range(len(self.net.bus)):
            measurements[f'V_{bus_idx+1}'] = self.net.res_bus.vm_pu[bus_idx]
            measurements[f'Theta_{bus_idx+1}'] = self.net.res_bus.va_degree[bus_idx]
            measurements[f'P_{bus_idx+1}'] = self.net.res_bus.p_mw[bus_idx]
            measurements[f'Q_{bus_idx+1}'] = self.net.res_bus.q_mvar[bus_idx]
        
        # 线路测量
        for idx, line in self.net.line.iterrows():
            key = f"L{int(line.from_bus)+1}-{int(line.to_bus)+1}"
            measurements[f'P_{key}'] = self.net.res_line.p_from_mw[idx]
            measurements[f'Q_{key}'] = self.net.res_line.q_from_mw[idx]
            measurements[f'Loading_{key}'] = self.net.res_line.loading_percent[idx]
        
        return measurements
    
    def _collect_dc_measurements(self):
        """收集DC测量数据（只有有功和相角）"""
        measurements = {}
        
        # 节点测量
        for bus_idx in range(len(self.net.bus)):
            measurements[f'Theta_{bus_idx+1}'] = self.net.res_bus.va_degree[bus_idx]
            measurements[f'P_{bus_idx+1}'] = self.net.res_bus.p_mw[bus_idx]
            # DC模型通常忽略无功和电压幅值变化
        
        # 线路测量（只有有功）
        for idx, line in self.net.line.iterrows():
            key = f"L{int(line.from_bus)+1}-{int(line.to_bus)+1}"
            measurements[f'P_{key}'] = self.net.res_line.p_from_mw[idx]
        
        return measurements
    
    def generate_simple_attack(self, normal_data, target_buses, attack_strength=0.1):
        """
        生成简单攻击（直接在测量值上添加扰动）
        
        参数:
        normal_data: 正常测量数据
        target_buses: 目标节点列表 [bus_index]
        attack_strength: 攻击强度 (0.1 = 10%)
        
        返回:
        受攻击的数据
        """
        attacked_data = normal_data.copy()
        
        for bus_idx in target_buses:
            # 攻击节点注入功率
            p_key = f'P_{bus_idx+1}'
            if p_key in attacked_data:
                base_value = attacked_data[p_key]
                # 随机选择攻击方向（增加或减少）
                direction = np.random.choice([-1, 1])
                attacked_data[p_key] = base_value * (1 + direction * attack_strength)
            
            # 攻击节点电压（如果需要）
            v_key = f'V_{bus_idx+1}'
            if v_key in attacked_data:
                base_value = attacked_data[v_key]
                attacked_data[v_key] = base_value * (1 + np.random.uniform(-0.01, 0.01))
        
        return attacked_data
    
    def generate_acdc_comparison_dataset(self, n_samples=100):
        """
        生成包含AC和DC潮流结果的对比数据集
        
        参数:
        n_samples: 样本数量
        
        返回:
        包含AC和DC结果的数据集
        """
        dataset = []
        
        for i in range(n_samples):
            # 随机决定是否断开线路
            line_outage = None
            if np.random.random() < 0.1:  # 10%概率断开线路
                # 随机选择一条线路
                line_idx = np.random.randint(0, len(self.net.line))
                line = self.net.line.iloc[line_idx]
                line_outage = [(line.from_bus, line.to_bus)]
            
            # 生成正常样本
            # AC潮流
            ac_measurements = self.run_ac_power_flow(
                load_variation=0.1,  # ±10%负荷波动
                line_outage=line_outage
            )
            
            # DC潮流
            dc_measurements = self.run_dc_power_flow(
                load_variation=0.1,
                line_outage=line_outage
            )
            
            if ac_measurements and dc_measurements:
                # 随机决定是否攻击
                is_attack = np.random.random() < 0.5
                
                if is_attack:
                    # 随机选择目标节点
                    target_buses = np.random.choice(
                        self.attackable_nodes, 
                        size=np.random.randint(1, 3),  # 1-2个节点
                        replace=False
                    )
                    
                    # 攻击强度
                    attack_strength = np.random.uniform(0.05, 0.2)  # 5-20%
                    
                    # 生成攻击数据
                    ac_attacked = self.generate_simple_attack(
                        ac_measurements, target_buses, attack_strength
                    )
                    dc_attacked = self.generate_simple_attack(
                        dc_measurements, target_buses, attack_strength
                    )
                    
                    sample = {
                        'sample_id': i,
                        'label': 1,  # 攻击
                        'attack_strength': attack_strength,
                        'target_buses': [int(b) for b in target_buses],
                        'line_outage': line_outage,
                        'ac_normal': ac_measurements,
                        'ac_attacked': ac_attacked,
                        'dc_normal': dc_measurements,
                        'dc_attacked': dc_attacked,
                    }
                else:
                    sample = {
                        'sample_id': i,
                        'label': 0,  # 正常
                        'attack_strength': 0,
                        'target_buses': [],
                        'line_outage': line_outage,
                        'ac_normal': ac_measurements,
                        'ac_attacked': None,
                        'dc_normal': dc_measurements,
                        'dc_attacked': None,
                    }
                
                dataset.append(sample)
        
        print(f"数据集生成完成! 总样本数: {len(dataset)}")
        print(f"正常样本: {sum(1 for s in dataset if s['label']==0)}")
        print(f"攻击样本: {sum(1 for s in dataset if s['label']==1)}")
        
        return dataset
    
    def compare_ac_dc_results(self, dataset, n_samples=5):
        """
        比较AC和DC潮流结果的差异
        
        参数:
        dataset: 生成的数据集
        n_samples: 展示的样本数量
        """
        print("\n=== AC与DC潮流结果对比 ===")
        
        for i, sample in enumerate(dataset[:n_samples]):
            print(f"\n样本 {i+1}: {'攻击' if sample['label']==1 else '正常'}")
            
            # 比较节点电压/相角
            print("\n节点1-5的电压/相角对比:")
            print(f"{'节点':<5} {'AC电压(pu)':<12} {'AC相角(度)':<12} {'DC相角(度)':<12}")
            print("-" * 50)
            
            for bus in range(5):
                ac_v = sample['ac_normal'].get(f'V_{bus+1}', 'N/A')
                ac_theta = sample['ac_normal'].get(f'Theta_{bus+1}', 'N/A')
                dc_theta = sample['dc_normal'].get(f'Theta_{bus+1}', 'N/A')
                
                if isinstance(ac_v, float):
                    print(f"{bus+1:<5} {ac_v:<12.4f} {ac_theta:<12.2f} {dc_theta:<12.2f}")
                else:
                    print(f"{bus+1:<5} {str(ac_v):<12} {str(ac_theta):<12} {str(dc_theta):<12}")
            
            # 比较线路潮流
            print("\n线路潮流对比 (MW):")
            line_keys = [k for k in sample['ac_normal'].keys() if k.startswith('P_L')]
            for key in line_keys[:3]:  # 只显示前3条线路
                ac_p = sample['ac_normal'].get(key, 0)
                dc_p = sample['dc_normal'].get(key, 0)
                if isinstance(ac_p, float) and isinstance(dc_p, float):
                    diff_pct = abs(ac_p - dc_p) / abs(ac_p) * 100 if ac_p != 0 else 0
                    print(f"{key:<10} AC: {ac_p:>7.2f} MW, DC: {dc_p:>7.2f} MW, 差异: {diff_pct:>5.1f}%")
    
    def visualize_ac_dc_differences(self, dataset):
        """可视化AC和DC结果的差异"""
        if not dataset:
            print("数据集为空!")
            return
        
        # 提取所有样本的AC和DC差异
        voltage_diffs = []
        angle_diffs = []
        power_diffs = []
        
        for sample in dataset:
            if sample['label'] == 0:  # 只使用正常样本
                # 计算电压差异（如果有）
                for bus in range(len(self.net.bus)):
                    ac_theta = sample['ac_normal'].get(f'Theta_{bus+1}')
                    dc_theta = sample['dc_normal'].get(f'Theta_{bus+1}')
                    
                    if isinstance(ac_theta, float) and isinstance(dc_theta, float):
                        angle_diffs.append(abs(ac_theta - dc_theta))
                
                # 计算线路潮流差异
                for key in sample['ac_normal'].keys():
                    if key.startswith('P_L'):
                        ac_p = sample['ac_normal'].get(key, 0)
                        dc_p = sample['dc_normal'].get(key, 0)
                        if isinstance(ac_p, float) and isinstance(dc_p, float) and ac_p != 0:
                            diff_pct = abs(ac_p - dc_p) / abs(ac_p) * 100
                            power_diffs.append(diff_pct)
        
        # 绘制差异分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 相角差异
        if angle_diffs:
            axes[0].hist(angle_diffs, bins=20, alpha=0.7, color='blue')
            axes[0].set_xlabel('相角差异绝对值 (度)')
            axes[0].set_ylabel('频次')
            axes[0].set_title('AC-DC相角差异分布')
            axes[0].grid(True, alpha=0.3)
        
        # 功率差异百分比
        if power_diffs:
            axes[1].hist(power_diffs, bins=20, alpha=0.7, color='red')
            axes[1].set_xlabel('线路潮流差异百分比 (%)')
            axes[1].set_ylabel('频次')
            axes[1].set_title('AC-DC线路潮流差异分布')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        if angle_diffs:
            print(f"相角差异统计:")
            print(f"  平均值: {np.mean(angle_diffs):.2f} 度")
            print(f"  最大值: {np.max(angle_diffs):.2f} 度")
        
        if power_diffs:
            print(f"线路潮流差异统计:")
            print(f"  平均差异: {np.mean(power_diffs):.1f}%")
            print(f"  最大差异: {np.max(power_diffs):.1f}%")

# 使用示例
def main():
    """主程序示例"""
    
    # 1. 初始化数据生成器
    print("初始化IEEE 14总线系统...")
    gen = SimpleFDIA_DataGenerator()
    
    # 2. 运行单次AC和DC潮流对比
    print("\n运行单次AC和DC潮流对比...")
    ac_result = gen.run_ac_power_flow(load_variation=0.1)
    dc_result = gen.run_dc_power_flow(load_variation=0.1)
    
    print("AC潮流节点1-3结果:")
    for key in ['V_1', 'Theta_1', 'P_1', 'Q_1', 'V_2', 'Theta_2', 'P_2']:
        if key in ac_result:
            print(f"  {key}: {ac_result[key]:.4f}")
    
    print("\nDC潮流节点1-3结果:")
    for key in ['Theta_1', 'P_1', 'Theta_2', 'P_2']:
        if key in dc_result:
            print(f"  {key}: {dc_result[key]:.4f}")
    
    # 3. 生成完整数据集
    print("\n生成完整数据集...")
    dataset = gen.generate_acdc_comparison_dataset(n_samples=50)  # 小规模测试
    
    # 4. 比较AC和DC结果
    gen.compare_ac_dc_results(dataset, n_samples=3)
    
    # 5. 可视化差异
    gen.visualize_ac_dc_differences(dataset)
    
    # 6. 保存数据集
    with open('acdc_fdia_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\n数据集已保存到 'acdc_fdia_dataset.pkl'")
    
    # 7. 显示攻击样本示例
    print("\n攻击样本示例:")
    attack_samples = [s for s in dataset if s['label'] == 1]
    if attack_samples:
        sample = attack_samples[0]
        print(f"攻击强度: {sample['attack_strength']*100:.1f}%")
        print(f"目标节点: {[b+1 for b in sample['target_buses']]}")
        
        # 显示攻击前后差异
        target = sample['target_buses'][0] if sample['target_buses'] else 2
        p_key = f'P_{target+1}'
        if p_key in sample['ac_normal']:
            ac_before = sample['ac_normal'][p_key]
            ac_after = sample['ac_attacked'][p_key]
            print(f"AC潮流 {p_key} 攻击前后:")
            print(f"  攻击前: {ac_before:.2f} MW")
            print(f"  攻击后: {ac_after:.2f} MW")
            print(f"  变化量: {abs(ac_after - ac_before):.2f} MW ({abs((ac_after - ac_before)/ac_before)*100:.1f}%)")

if __name__ == "__main__":
    main()