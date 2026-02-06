import warnings
warnings.filterwarnings('ignore', message='numba cannot be imported')

import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑、黑体
plt.rcParams['axes.unicode_minus'] = False

class PowerSystemDataGenerator:
    """
    稳定版本的电力系统数据生成器
    """
    
    def __init__(self, sampling_rate=10, total_hours=24):
        self.sampling_rate = sampling_rate
        self.total_hours = total_hours
        self.total_samples = total_hours * 3600 * sampling_rate
        self.base_net = None
        self.net = None
        self.measurement_dim = 56
        self.data_history = []
        self.base_loads = {}  # 添加这一行：初始化base_loads
        
    def create_ieee14_network(self):
        """创建IEEE14系统，使用标准测试系统参数"""
        self.net = nw.case14()
        
        # IEEE14标准参数（来自MATPOWER/IEEE标准测试系统）
        # 发电机参数
        gen_data = [
            # bus, Pg(MW), Qg(MVar), Vset(p.u.)
            (1, 232.4, -16.9, 1.060),   # Slack bus
            (2, 40.0, 50.0, 1.045),     # PV bus
            (3, 0.0, 23.4, 1.010),      # PV bus
            (6, 0.0, 12.2, 1.070),      # PV bus
            (8, 0.0, 17.4, 1.090)       # PV bus
        ]
        
        # 更新发电机参数
        for i, gen_idx in enumerate(self.net.gen.index):
            if i < len(gen_data):  # 添加边界检查
                bus, Pg, Qg, Vset = gen_data[i]
                self.net.gen.at[gen_idx, 'vm_pu'] = Vset
                self.net.gen.at[gen_idx, 'p_mw'] = Pg
                self.net.gen.at[gen_idx, 'q_mvar'] = Qg
        
        # 负载参数（IEEE14标准）
        load_data = [
            # bus, Pd(MW), Qd(MVar)
            (2, 21.7, 12.7),
            (3, 94.2, 19.0),
            (4, 47.8, -3.9),
            (5, 7.6, 1.6),
            (6, 11.2, 7.5),
            (9, 29.5, 16.6),
            (10, 9.0, 5.8),
            (11, 3.5, 1.8),
            (12, 6.1, 1.6),
            (13, 13.5, 5.8),
            (14, 14.9, 5.0)
        ]
        
        # 更新负载参数并保存到base_loads
        for bus, Pd, Qd in load_data:
            # 找到该总线上的负载索引
            load_idx = self.net.load[self.net.load.bus == bus].index
            if len(load_idx) > 0:
                load_idx = load_idx[0]
                self.net.load.at[load_idx, 'p_mw'] = Pd
                self.net.load.at[load_idx, 'q_mvar'] = Qd
                # 保存基准负荷值到base_loads
                self.base_loads[load_idx] = {
                    'p_mw': Pd,
                    'q_mvar': Qd
                }
        
        print("IEEE14系统创建完成，使用标准参数")
        print(f"已保存基准负荷: {len(self.base_loads)}个负载节点")
        return self.net
    
    def add_random_variation(self, base_value, variation_type="load", hour_of_day=12):
        """
        添加随机变化到基准值
        
        参数:
        base_value: 基准值
        variation_type: 变化类型 ("load", "generation")
        hour_of_day: 一天中的小时数 (0-24)
        """
        # 基于小时生成日负荷曲线因子
        t = hour_of_day / 24 * 2 * np.pi
        
        if variation_type == "load":
            # 负载变化：日负荷曲线 + 随机波动
            daily_factor = 0.7 + 0.3 * (np.sin(t - np.pi/2) + 0.3 * np.sin(2*t)) / 1.3
            random_factor = 1 + np.random.normal(0, 0.05)  # ±5%随机波动
            variation_factor = daily_factor * random_factor
            
        elif variation_type == "generation":
            # 发电变化：相对稳定，轻微波动
            daily_factor = 0.8 + 0.2 * np.sin(t) / 1.0
            random_factor = 1 + np.random.normal(0, 0.02)  # ±2%随机波动
            variation_factor = daily_factor * random_factor
        
        else:
            variation_factor = 1.0
        
        # 限制变化范围
        variation_factor = np.clip(variation_factor, 0.8, 1.2)
        
        return base_value * variation_factor
    
    def run_power_flow(self, timestamp):
        """
        简化的潮流计算函数
        
        只包含核心功能：
        1. 基于时间调整负载
        2. 执行潮流计算
        3. 返回测量值
        """
        # 检查base_loads是否已初始化
        if not self.base_loads:
            # 如果base_loads为空，从当前网络初始化
            print("警告: base_loads未初始化，从当前网络获取基准值")
            for load_idx in self.net.load.index:
                self.base_loads[load_idx] = {
                    'p_mw': self.net.load.at[load_idx, 'p_mw'],
                    'q_mvar': self.net.load.at[load_idx, 'q_mvar']
                }
        
        # 1. 计算时间相关因子
        hour_of_day = timestamp.hour + timestamp.minute / 60.0
        hour_int = int(hour_of_day) % 24
        
        # 简单日负荷曲线
        day_profile = [0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 
                    0.9, 1.0, 0.95, 0.9, 0.85, 0.8, 0.8, 0.85,
                    0.9, 0.95, 1.0, 1.0, 0.95, 0.9, 0.8, 0.7]
        
        load_factor = day_profile[hour_int]
        
        # 2. 更新所有负载
        for load_idx in self.net.load.index:
            if load_idx in self.base_loads:
                base_p = self.base_loads[load_idx]['p_mw']
                base_q = self.base_loads[load_idx]['q_mvar']
                
                # 添加随机波动
                random_factor = 1 + np.random.normal(0, 0.05)  # ±5%波动
                current_factor = load_factor * random_factor
                
                self.net.load.at[load_idx, 'p_mw'] = base_p * current_factor
                self.net.load.at[load_idx, 'q_mvar'] = base_q * current_factor
        
        # 3. 执行潮流计算
        try:
            pp.runpp(self.net)
            return self.extract_measurements()
        except Exception as e:
            print(f"潮流计算失败: {e}")
            # 返回保守估计
            return self._generate_conservative_estimate(timestamp)
    
    def extract_measurements(self):
        """从网络提取测量值"""
        if self.net is None:
            print("警告: 网络未初始化")
            return np.zeros(self.measurement_dim, dtype=np.float32)
        
        measurements = np.zeros(self.measurement_dim, dtype=np.float32)
        
        try:
            # 电压幅值 (14个)
            for i, bus_idx in enumerate(sorted(self.net.bus.index)):
                measurements[i] = self.net.res_bus.at[bus_idx, 'vm_pu']
            
            # 电压相角 (14个)
            for i, bus_idx in enumerate(sorted(self.net.bus.index)):
                measurements[14 + i] = self.net.res_bus.at[bus_idx, 'va_degree']
            
            # 有功功率 (14个)
            for i, bus_idx in enumerate(sorted(self.net.bus.index)):
                measurements[28 + i] = self.net.res_bus.at[bus_idx, 'p_mw']
            
            # 无功功率 (14个)
            for i, bus_idx in enumerate(sorted(self.net.bus.index)):
                measurements[42 + i] = self.net.res_bus.at[bus_idx, 'q_mvar']
                
        except Exception as e:
            print(f"提取测量值时出错: {e}")
            # 返回保守估计
            if self.data_history:
                return np.mean(self.data_history[-5:], axis=0)
            else:
                return np.zeros(self.measurement_dim, dtype=np.float32)
        
        return measurements
    
    def get_measurement_dimension(self):
        """获取测量维度"""
        return self.measurement_dim
    
    def _generate_conservative_estimate(self, timestamp):
        """保守估计：返回历史均值或基准值"""
        if self.data_history:
            # 返回最近的历史平均值
            if len(self.data_history) >= 10:
                return np.mean(self.data_history[-10:], axis=0)
            else:
                return self.data_history[-1]
        else:
            # 返回基准值
            return np.array([1.0] * 14 + [0.0] * 14 + [0.0] * 28, dtype=np.float32)
    
    def generate_normal_data(self, save_path=None):
        """
        生成正常状态下的数据
        
        参数:
        save_path: 数据保存路径
        
        返回:
        data: 正常数据数组 (n_samples, n_features)
        timestamps: 时间戳列表
        """
        print("开始生成正常数据...")
        
        if self.net is None:
            self.create_ieee14_network()
        
        # 计算总样本数
        total_samples = int(self.total_samples)
        
        # 初始化数据数组
        data = np.zeros((total_samples, self.measurement_dim), dtype=np.float32)
        timestamps = []
        
        # 起始时间
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 生成每个时间步的数据
        for i in range(total_samples):
            # 计算当前时间戳
            current_time = start_time + timedelta(seconds=i/self.sampling_rate)
            timestamps.append(current_time)
            
            # 运行潮流计算获取测量值
            measurements = self.run_power_flow(current_time)
            data[i] = measurements
            self.data_history.append(measurements)
            
            # 显示进度
            # 每20%打印一次进度
            percentage_interval = 0.2  
            if i % int(total_samples * percentage_interval) == 0 and i > 0:
                percent = (i / total_samples) * 100
                hour = i / (3600 * self.sampling_rate)
                print(f"  进度: {percent:.1f}% ({hour:.2f}小时 / {self.total_hours:.1f}小时)")
        
        print(f"正常数据生成完成: {data.shape}")
        
        # 保存数据
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamps': timestamps,
                    'sampling_rate': self.sampling_rate,
                    'description': 'IEEE14系统正常状态数据'
                }, f)
            print(f"数据保存到 {save_path}")
        
        return data, timestamps


class FDIA_DataPipeline:
    """
    FDIA数据处理完整流水线
    集成数据生成、攻击注入、滑动窗口处理
    """
    
    def __init__(self, config=None):
        """
        配置参数：
        config: 配置字典，包含所有参数
        """
        if config is None:
            config = {
                'sampling_rate': 10,  # Hz
                'total_hours': 2,     # 小时
                'window_size': 10,    # 时间步
                'step': 2,           # 滑动步长
                'n_attacks': 10,     # 攻击数量
                'min_duration': 30,  # 最小攻击持续时间
                'max_duration': 100, # 最大攻击持续时间
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'use_feature_enhancement': True,  # 新增：是否使用特征扩展
                'feature_dim': 56                 # 新增：特征维度（扩展后会更新）
            }
        
        self.config = config
        self.generator = None
        self.injector = None
        self.processor = None
        self.normalizer = None
        
        # 数据存储
        self.normal_data = None
        self.attacked_data = None
        self.labels = None
        self.attack_infos = None
        
        # 窗口数据
        self.X_windows = None
        self.y_windows = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    def plot_sample_data(self, data, start_idx=0, duration=200, save_path=None):
        """
        可视化样本数据
        
        参数:
        data: 数据数组
        start_idx: 起始索引
        duration: 持续时间步数
        save_path: 保存路径
        """
        end_idx = min(start_idx + duration, len(data))
        sample_data = data[start_idx:end_idx]
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. 电压幅值 (前14维)
        for i in range(min(5, 14)):
            axes[0].plot(sample_data[:, i], label=f'Bus {i+1}')
        axes[0].set_title('电压幅值 (pu)')
        axes[0].set_ylabel('电压 (pu)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')
        
        # 2. 电压相角 (14-27维)
        for i in range(min(5, 14)):
            axes[1].plot(sample_data[:, 14 + i], label=f'Bus {i+1}')
        axes[1].set_title('电压相角 (度)')
        axes[1].set_ylabel('相角 (度)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        
        # 3. 有功功率 (28-41维)
        for i in range(min(5, 14)):
            axes[2].plot(sample_data[:, 28 + i], label=f'Bus {i+1}')
        axes[2].set_title('有功功率 (MW)')
        axes[2].set_ylabel('功率 (MW)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
        
        # 4. 无功功率 (42-55维)
        for i in range(min(5, 14)):
            axes[3].plot(sample_data[:, 42 + i], label=f'Bus {i+1}')
        axes[3].set_title('无功功率 (MVar)')
        axes[3].set_xlabel('时间步')
        axes[3].set_ylabel('功率 (MVar)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"样本可视化保存到 {save_path}")
        
        plt.show()


    def run_full_pipeline(self, attack_type=None, regenerate=False):
        """
        运行完整数据处理流水线
        
        参数：
        regenerate: 是否重新生成数据
        """
        print("=" * 60)
        print("FDIA数据处理流水线启动")
        print("=" * 60)
        
        # 1. 生成正常数据
        normal_data_path = "data/normal_data.pkl"
        
        if regenerate or not os.path.exists(normal_data_path):
            print("\n[阶段1] 生成正常数据...")
            # 使用稳定版本
            self.generator = PowerSystemDataGenerator(
                sampling_rate=self.config['sampling_rate'],
                total_hours=self.config['total_hours']
            )
            self.normal_data, timestamps = self.generator.generate_normal_data(normal_data_path)
            
            # 可视化样本
            self.plot_sample_data(
                self.normal_data, 
                start_idx=0, 
                duration=200,
                save_path="figures/normal_data_sample.png"
            )
        else:
            print("\n[阶段1] 加载已有正常数据...")
            with open(normal_data_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.normal_data = data_dict['data']
            
            print(f"正常数据加载完成: {self.normal_data.shape}")
        
        # 2. 注入攻击
        attack_data_path = "data/attack_data.pkl"
        
        if regenerate or not os.path.exists(attack_data_path):
            print("\n[阶段2] 注入FDIA攻击...")
            self.injector = FDIAAttackInjector(self.normal_data)
            if attack_type is None:
                # 混合攻击（原逻辑）
                attack_data_path = "data/attack_data_mixed.pkl"
                if regenerate or not os.path.exists(attack_data_path):
                    print("\n[阶段2] 注入混合FDIA攻击...")
                    self.injector = FDIAAttackInjector(self.normal_data)
                    self.attacked_data, self.labels, self.attack_infos = self.injector.generate_attack_dataset(
                        n_attacks=self.config['n_attacks'],
                        min_duration=self.config['min_duration'],
                        max_duration=self.config['max_duration'],
                        save_path=attack_data_path
                    )
            else:
                # 特定类型攻击
                attack_data_path = f"data/attack_data_{attack_type}.pkl"
                if regenerate or not os.path.exists(attack_data_path):
                    print(f"\n[阶段2] 注入{attack_type}类型FDIA攻击...")
                    self.injector = FDIAAttackInjector(self.normal_data)
                    # 使用新的单类型攻击生成方法
                    self.attacked_data, self.labels, self.attack_infos = self.injector.generate_specific_attack_dataset(
                        attack_type=attack_type,
                        n_attacks=self.config['n_attacks'],
                        min_duration=self.config['min_duration'],
                        max_duration=self.config['max_duration'],
                        save_path_template=f"data/attack_{{type}}.pkl"
                    )
            
            # 可视化攻击
            self.injector.visualize_attacks(
                self.attacked_data, 
                self.labels, 
                self.attack_infos,
                save_path="figures/attack_visualization.png"
            )
        else:
            print("\n[阶段2] 加载已有攻击数据...")
            with open(attack_data_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.attacked_data = data_dict['data']
                self.labels = data_dict['labels']
                self.attack_infos = data_dict['attack_infos']
            
            print(f"攻击数据加载完成: {self.attacked_data.shape}")
            print(f"攻击比例: {np.sum(self.labels)/len(self.labels)*100:.2f}%")
        
        # 3. 创建滑动窗口
        print("\n[阶段3] 创建滑动窗口...")
        self.processor = SlidingWindowProcessor(
            window_size=self.config['window_size'],
            step=self.config['step'],
            feature_dim=self.generator.get_measurement_dimension()
        )
        
        self.X_windows, self.y_windows = self.processor.create_sliding_windows(
            self.attacked_data, 
            self.labels
        )
        
        print(f"滑动窗口创建完成: {self.X_windows.shape}")
        
        # 分析类别不平衡
        self.processor.analyze_class_imbalance(self.y_windows)
        
        # 可视化窗口样本
        self.processor.visualize_windows(
            self.X_windows, 
            self.y_windows,
            n_samples=3,
            save_path="figures/window_samples.png"
        )
        
        # 4. 划分数据集
        print("\n[阶段4] 划分训练/验证/测试集...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self.processor.create_dataset_splits(
                self.X_windows, 
                self.y_windows,
                train_ratio=self.config['train_ratio'],
                val_ratio=self.config['val_ratio']
            )
        
        if self.config.get('use_feature_enhancement', True):
            print("\n[阶段4.5] 特征扩展（56维 → 84维）...")
            self.enhancer = FeatureEnhancer84D()
            
            # 扩展训练集
            print(f"  扩展训练集: {X_train.shape} → ", end="")
            X_train = self.enhancer.enhance_batch(X_train)
            print(f"{X_train.shape}")
            
            # 扩展验证集
            print(f"  扩展验证集: {X_val.shape} → ", end="")
            X_val = self.enhancer.enhance_batch(X_val)
            print(f"{X_val.shape}")
            
            # 扩展测试集
            print(f"  扩展测试集: {X_test.shape} → ", end="")
            X_test = self.enhancer.enhance_batch(X_test)
            print(f"{X_test.shape}")
            
            # 更新特征维度配置
            self.config['feature_dim'] = X_train.shape[2]
            print(f"  特征维度更新: {self.config['feature_dim']}维")

        # 保存扩展后的X_windows（用于后续分析）
        self.X_windows = np.concatenate([X_train, X_val, X_test], axis=0)
        self.y_windows = np.concatenate([y_train, y_val, y_test], axis=0)

        # 5. 数据标准化
        print("\n[阶段5] 数据标准化...")
        self.X_train = X_train  # 更新为扩展后的数据
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        # 5. 数据标准化
        print("\n[阶段5] 数据标准化...")
        self.normalizer = DataNormalizer()
        self.X_train = self.normalizer.fit_transform(self.X_train)
        self.X_val = self.normalizer.transform(self.X_val)
        self.X_test = self.normalizer.transform(self.X_test)
        
        # 保存标准化参数
        self.normalizer.save("models/normalizer.pkl")
        
        # 6. 保存处理后的数据集
        print("\n[阶段6] 保存最终数据集...")
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("数据处理流水线完成!")
        print("=" * 60)
        
        return self
    
    def save_processed_data(self, save_dir="processed_data"):
        """保存处理后的数据集"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为numpy格式
        np.save(f"{save_dir}/X_train.npy", self.X_train)
        np.save(f"{save_dir}/y_train.npy", self.y_train)
        np.save(f"{save_dir}/X_val.npy", self.X_val)
        np.save(f"{save_dir}/y_val.npy", self.y_val)
        np.save(f"{save_dir}/X_test.npy", self.X_test)
        np.save(f"{save_dir}/y_test.npy", self.y_test)
        
        # 保存配置信息
        import json
        with open(f"{save_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"处理后的数据保存到 {save_dir}/")
    
    def load_processed_data(self, load_dir="processed_data"):
        """加载处理后的数据集"""
        self.X_train = np.load(f"{load_dir}/X_train.npy")
        self.y_train = np.load(f"{load_dir}/y_train.npy")
        self.X_val = np.load(f"{load_dir}/X_val.npy")
        self.y_val = np.load(f"{load_dir}/y_val.npy")
        self.X_test = np.load(f"{load_dir}/X_test.npy")
        self.y_test = np.load(f"{load_dir}/y_test.npy")
        
        # 加载配置
        import json
        with open(f"{load_dir}/config.json", 'r') as f:
            self.config = json.load(f)
        
        print(f"处理后的数据从 {load_dir}/ 加载")
        print(f"训练集: {self.X_train.shape}, {self.y_train.shape}")
        print(f"验证集: {self.X_val.shape}, {self.y_val.shape}")
        print(f"测试集: {self.X_test.shape}, {self.y_test.shape}")
        
        return self
    
    def get_data_loaders(self, batch_size=32):
        """获取PyTorch数据加载器"""
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)  # 添加维度
        
        X_val_tensor = torch.FloatTensor(self.X_val)
        y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1)
        
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_test_tensor = torch.FloatTensor(self.y_test).unsqueeze(1)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"数据加载器创建完成:")
        print(f"  训练集: {len(train_loader)}批次, 批次大小={batch_size}")
        print(f"  验证集: {len(val_loader)}批次")
        print(f"  测试集: {len(test_loader)}批次")
        
        return train_loader, val_loader, test_loader
    
    def generate_all_attack_types(self,num=20):
        """生成所有类型的攻击数据集"""
        attack_types = ['single_point', 'multi_point', 'slow_drift', 
                    'physical_constrained', 'random_noise']
        
        for attack_type in attack_types:
            print(f"\n{'='*60}")
            print(f"生成 {attack_type} 攻击数据集")
            print(f"{'='*60}")
            
            # 临时创建攻击注入器
            injector = FDIAAttackInjector(self.normal_data)
            
            # 生成并保存该类型攻击数据集
            data, labels, infos = injector.generate_specific_attack_dataset(
                attack_type=attack_type,
                n_attacks=num,  # 每种攻击生成多个实例
                min_duration=30,
                max_duration=100,
                save_path_template=f"data/attack_{{type}}.pkl"
            )
            
            # 可视化
            injector.visualize_attacks(
                data, labels, infos,
                save_path=f"figures/attack_{attack_type}.png"
            )
        
        print("\n所有攻击类型数据集生成完成！")

    def summarize(self):
        """打印数据流水线摘要"""
        print("\n" + "=" * 60)
        print("FDIA数据流水线摘要")
        print("=" * 60)
        
        print(f"\n1. 原始数据:")
        if self.normal_data is not None:
            print(f"   正常数据形状: {self.normal_data.shape}")
        if self.attacked_data is not None:
            print(f"   攻击数据形状: {self.attacked_data.shape}")
        
        print(f"\n2. 滑动窗口配置:")
        print(f"   窗口大小: {self.config['window_size']}")
        print(f"   滑动步长: {self.config['step']}")
        print(f"   特征维度: {self.config.get('feature_dim', 56)}")
        
        print(f"\n3. 处理后窗口数据:")
        print(f"   总窗口数: {len(self.X_windows) if self.X_windows is not None else 'N/A'}")
        
        print(f"\n4. 数据集划分:")
        if self.X_train is not None:
            print(f"   训练集: {self.X_train.shape}")
            print(f"   验证集: {self.X_val.shape}")
            print(f"   测试集: {self.X_test.shape}")
            
            # 类别分布
            train_attack_ratio = np.sum(self.y_train)/len(self.y_train)*100
            val_attack_ratio = np.sum(self.y_val)/len(self.y_val)*100
            test_attack_ratio = np.sum(self.y_test)/len(self.y_test)*100
            
            print(f"\n5. 类别分布:")
            print(f"   训练集攻击比例: {train_attack_ratio:.2f}%")
            print(f"   验证集攻击比例: {val_attack_ratio:.2f}%")
            print(f"   测试集攻击比例: {test_attack_ratio:.2f}%")
        
        print("\n" + "=" * 60)


class DataNormalizer:
    """
    数据标准化处理器
    处理训练和测试数据的标准化
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X_train):
        """基于训练数据计算标准化参数"""
        # X_train形状: (n_samples, window_size, n_features)
        # 计算每个特征的均值和标准差
        self.mean = np.mean(X_train, axis=(0, 1))  # 按特征维度平均
        self.std = np.std(X_train, axis=(0, 1))
        
        # 防止除零
        self.std = np.where(self.std == 0, 1.0, self.std)
        
        self.fitted = True
        print(f"标准化参数计算完成: mean={self.mean.shape}, std={self.std.shape}")
        
        return self
    
    def transform(self, X):
        """应用标准化"""
        if not self.fitted:
            raise ValueError("必须先调用fit方法")
        
        # 标准化
        X_normalized = (X - self.mean) / self.std
        
        return X_normalized
    
    def fit_transform(self, X_train):
        """拟合并转换"""
        return self.fit(X_train).transform(X_train)
    
    def inverse_transform(self, X_normalized):
        """反向标准化"""
        if not self.fitted:
            raise ValueError("必须先调用fit方法")
        
        X_original = X_normalized * self.std + self.mean
        
        return X_original
    
    def save(self, path="models/normalizer.pkl"):
        """保存标准化参数"""
        import pickle
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'mean': self.mean,
                'std': self.std
            }, f)
        print(f"标准化参数保存到 {path}")
    
    def load(self, path="models/normalizer.pkl"):
        """加载标准化参数"""
        import pickle
        
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = True
        print(f"标准化参数从 {path} 加载")
        
        return self
    

# 84维特征扩展
class FeatureEnhancer84D:
    """
    将56维IEEE14特征扩展到84维
    """
    
    def __init__(self):
        # IEEE14标准拓扑连接关系
        self.connections = [
            (0,1), (0,4), (1,2), (1,3), (1,4), (2,3), (3,4), 
            (3,6), (3,8), (4,5), (5,10), (5,11), (5,12), 
            (6,7), (6,8), (8,9), (8,13), (9,10), (11,12), (12,13)
        ]
    
    def enhance_batch(self, X_batch):
        """
        批量处理：将56维扩展到84维
        
        输入: X_batch (batch_size, window_size, 56)
        输出: X_enhanced (batch_size, window_size, 84)
        """
        batch_size, window_size, _ = X_batch.shape
        enhanced_batch = np.zeros((batch_size, window_size, 84))
        
        for b in range(batch_size):
            for t in range(window_size):
                enhanced_batch[b, t] = self.enhance_single(X_batch[b, t])
        
        return enhanced_batch
    
    def enhance_single(self, measurement):
        """
        单个时间步特征扩展
        """
        # 1. 原始特征 (0-55) - 56维
        enhanced = measurement.copy()
        
        # 提取原始特征
        vm = measurement[0:14]    # 电压幅值
        va = measurement[14:28]   # 电压相角
        p = measurement[28:42]    # 有功功率
        q = measurement[42:56]    # 无功功率
        
        # 2. 拓扑关系特征 (56-65: 10维)
        voltage_diffs = []
        angle_diffs = []
        
        for i, j in self.connections:
            voltage_diffs.append(abs(vm[i] - vm[j]))
            angle_diffs.append(abs(va[i] - va[j]))
        
        voltage_diff_avg = np.mean(voltage_diffs) if voltage_diffs else 0
        angle_diff_max = np.max(angle_diffs) if angle_diffs else 0
        
        total_p = np.sum(p)
        total_q = np.sum(q)
        sum_p_pos = np.sum(p[p > 0])
        sum_q_pos = np.sum(q[q > 0])
        p_imbalance = abs(total_p) / max(abs(sum_p_pos), 0.01) if sum_p_pos != 0 else 0
        q_imbalance = abs(total_q) / max(abs(sum_q_pos), 0.01) if sum_q_pos != 0 else 0
        
        key_buses = [0, 1, 2, 3, 4]
        key_powers = p[key_buses]
        key_voltages = vm[key_buses]
        
        # 修正：确保正好10个拓扑特征
        topology_features = np.array([
            voltage_diff_avg,                    # 1
            angle_diff_max,                      # 2
            p_imbalance,                         # 3
            q_imbalance,                         # 4
            np.mean(key_powers),                 # 5
            np.std(key_powers),                  # 6
            np.mean(key_voltages),               # 7
            np.std(key_voltages),                # 8
            len(self.connections),               # 9
            voltage_diff_avg / (angle_diff_max + 0.01)  # 10
        ])
        
        enhanced = np.concatenate([enhanced, topology_features])
        
        # 3. 物理约束特征 (66-73: 8维)
        vm_std = np.std(vm)
        vm_range = np.max(vm) - np.min(vm)
        va_std = np.std(va)
        va_range = np.max(va) - np.min(va)
        
        apparent_power = np.sqrt(np.sum(p)**2 + np.sum(q)**2)
        power_factor = abs(np.sum(p)) / apparent_power if apparent_power > 0 else 0
        
        gen_buses = [0, 1, 2, 5, 7]
        total_gen_p = np.sum(p[gen_buses])
        
        phys_features = np.array([
            vm_std,                              # 1
            vm_range,                            # 2
            va_std,                              # 3
            va_range,                            # 4
            power_factor,                        # 5
            np.sum(q) / max(abs(np.sum(p)), 0.01),  # 6
            total_gen_p / max(abs(np.sum(p)), 0.01),  # 7
            total_gen_p                          # 8
        ])

        enhanced = np.concatenate([enhanced, phys_features])
        
        # 4. 统计特征 (74-79: 6维)
        stat_features = np.array([
            np.mean(vm),    # 1
            np.std(vm),     # 2
            np.mean(p),     # 3
            np.std(p),      # 4
            np.mean(q),     # 5
            np.std(q)       # 6
        ])
        
        enhanced = np.concatenate([enhanced, stat_features])
        
        # 5. 归一化特征 (80-83: 4维)
        base_voltage = 1.0
        base_power = 100.0
        
        norm_features = np.array([
            np.mean(vm) / base_voltage,                     # 1
            np.max(np.abs(p)) / base_power,                 # 2
            np.sum(p[p>0]) / base_power,                    # 3
            np.sum(np.abs(p[p<0])) / base_power             # 4
        ])
        
        enhanced = np.concatenate([enhanced, norm_features])
        
        # 最终检查
        if len(enhanced) != 84:
            if len(enhanced) < 84:
                enhanced = np.concatenate([enhanced, np.zeros(84 - len(enhanced))])
            else:
                enhanced = enhanced[:84]
        
        return enhanced


class SlidingWindowProcessor:
    """
    滑动窗口处理器
    将时间序列数据转换为滑动窗口样本
    """
    
    def __init__(self, window_size=10, step=1, feature_dim=56):
        """
        参数：
        window_size: 窗口大小（时间步数）
        step: 滑动步长
        feature_dim: 特征维度
        """
        self.window_size = window_size
        self.step = step
        self.feature_dim = feature_dim
        
    def create_sliding_windows(self, data, labels=None):
        """
        创建滑动窗口样本
        
        参数：
        data: 时间序列数据 (n_samples, n_features)
        labels: 标签数据 (n_samples,) 或 (n_samples, n_features)
        
        返回：
        X_windows: 窗口数据 (n_windows, window_size, n_features)
        y_windows: 窗口标签 (n_windows,) 或 (n_windows, window_size, n_features)
        """
        n_samples = len(data)
        
        # 计算窗口数量
        n_windows = (n_samples - self.window_size) // self.step + 1
        
        # 初始化窗口数组
        X_windows = np.zeros((n_windows, self.window_size, self.feature_dim), dtype=np.float32)
        
        # 填充窗口
        for i in range(n_windows):
            start_idx = i * self.step
            end_idx = start_idx + self.window_size
            X_windows[i] = data[start_idx:end_idx]
        
        # 处理标签
        if labels is not None:
            if labels.ndim == 1:  # 检测任务：每个窗口一个标签
                # 窗口标签 = 窗口内最后一个时间步的标签
                y_start = (self.window_size - 1)
                y_indices = range(y_start, n_samples, self.step)
                y_windows = labels[y_indices][:n_windows]
                return X_windows, y_windows
            
            elif labels.ndim == 2:  # 定位任务：每个窗口每个时间步都有标签
                y_windows = np.zeros((n_windows, self.window_size, self.feature_dim), dtype=np.int32)
                for i in range(n_windows):
                    start_idx = i * self.step
                    end_idx = start_idx + self.window_size
                    y_windows[i] = labels[start_idx:end_idx]
                return X_windows, y_windows
        
        return X_windows
    
    def create_dataset_splits(self, X_windows, y_windows, train_ratio=0.7, val_ratio=0.15):
        """
        划分训练集、验证集和测试集
        
        参数：
        X_windows: 窗口特征
        y_windows: 窗口标签
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
        返回：
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        n_samples = len(X_windows)
        
        # 计算划分点
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 划分数据集
        X_train = X_windows[:train_end]
        y_train = y_windows[:train_end]
        
        X_val = X_windows[train_end:val_end]
        y_val = y_windows[train_end:val_end]
        
        X_test = X_windows[val_end:]
        y_test = y_windows[val_end:]
        
        # 打印统计信息
        print("\n数据集划分统计:")
        print(f"总窗口数: {n_samples}")
        print(f"训练集: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
        print(f"验证集: {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
        print(f"测试集: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
        
        if y_windows.ndim == 1:
            print(f"\n各类别数量:")
            for split_name, X_split, y_split in [
                ('训练集', X_train, y_train),
                ('验证集', X_val, y_val),
                ('测试集', X_test, y_test)
            ]:
                n_normal = np.sum(y_split == 0)
                n_attack = np.sum(y_split == 1)
                print(f"{split_name}: 正常={n_normal}, 攻击={n_attack} (攻击比例={n_attack/len(y_split)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def analyze_class_imbalance(self, y_windows):
        """分析类别不平衡情况"""
        n_normal = np.sum(y_windows == 0)
        n_attack = np.sum(y_windows == 1)
        total = len(y_windows)
        
        print("\n类别不平衡分析:")
        print(f"正常样本: {n_normal} ({n_normal/total*100:.2f}%)")
        print(f"攻击样本: {n_attack} ({n_attack/total*100:.2f}%)")
        print(f"不平衡比例: {n_normal/max(n_attack, 1):.2f}:1")
        
        return n_normal, n_attack
    
    def visualize_windows(self, X_windows, y_windows, n_samples=5, save_path=None):
        """可视化滑动窗口样本"""
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        # 随机选择几个窗口
        indices = np.random.choice(len(X_windows), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            window_data = X_windows[idx]
            window_label = y_windows[idx] if y_windows.ndim == 1 else np.any(y_windows[idx])
            
            # 左侧：窗口内所有特征
            axes[i, 0].imshow(window_data.T, aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'窗口 {idx}: 特征随时间变化 (标签: {window_label})')
            axes[i, 0].set_xlabel('时间步')
            axes[i, 0].set_ylabel('特征索引')
            
            # 右侧：几个代表性特征的时间序列
            sample_features = [0, 14, 28, 42]  # 总线0的各个测量值
            for feat_idx in sample_features:
                axes[i, 1].plot(window_data[:, feat_idx], label=f'特征{feat_idx}')
            
            axes[i, 1].set_title(f'窗口 {idx}: 关键特征时间序列')
            axes[i, 1].set_xlabel('窗口内时间步')
            axes[i, 1].set_ylabel('测量值')
            axes[i, 1].legend(loc='upper right', fontsize='small')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"窗口可视化保存到 {save_path}")
        
        plt.show()


class FDIAAttackInjector:
    """
    虚假数据注入攻击（FDIA）生成器
    在正常数据基础上注入各种类型的攻击
    """
    
    def __init__(self, normal_data):
        """
        参数：
        normal_data: 正常数据数组 (n_samples, n_features)
        """
        self.normal_data = normal_data
        self.n_samples, self.n_features = normal_data.shape
        
        # 攻击类型定义
        self.attack_types = {
            'single_point': '单点突增攻击',
            'multi_point': '多点协同攻击',
            'slow_drift': '缓慢漂移攻击',
            'physical_constrained': '物理约束攻击',
            'random_noise': '随机噪声攻击'
        }
    
    def inject_single_point_attack(self, start_idx, duration, target_feature, magnitude=0.1):
        """
        注入单点突增攻击
        
        参数：
        start_idx: 攻击开始位置
        duration: 攻击持续时间（采样点数）
        target_feature: 目标特征索引（0-55）
        magnitude: 攻击强度（相对于正常值的比例）
        """
        attacked_data = self.normal_data.copy()
        
        # 计算攻击值（突增）
        end_idx = min(start_idx + duration, self.n_samples)
        original_values = attacked_data[start_idx:end_idx, target_feature]
        
        # 攻击：在原始值上增加固定百分比
        attack_values = original_values * (1 + magnitude)
        attacked_data[start_idx:end_idx, target_feature] = attack_values
        
        # 创建攻击标签（1表示攻击，0表示正常）
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        
        # 攻击详情
        attack_info = {
            'type': 'single_point',
            'start_idx': start_idx,
            'duration': duration,
            'target_feature': target_feature,
            'magnitude': magnitude,
            'description': f'单点攻击：特征{target_feature}在[{start_idx}:{end_idx}]增加{magnitude*100}%'
        }
        
        return attacked_data, attack_labels, attack_info
    
    def inject_multi_point_attack(self, start_idx, duration, target_features, correlation=0.8):
        """
        注入多点协同攻击
        
        参数：
        start_idx: 攻击开始位置
        duration: 攻击持续时间
        target_features: 目标特征列表
        correlation: 特征间相关性（0-1）
        """
        attacked_data = self.normal_data.copy()
        
        end_idx = min(start_idx + duration, self.n_samples)
        n_targets = len(target_features)
        
        # 生成相关攻击信号
        attack_signal = np.random.normal(0, 0.3, size=(end_idx - start_idx, n_targets))
        
        # 添加相关性
        if correlation > 0:
            # 创建相关矩阵
            corr_matrix = np.eye(n_targets) * (1 - correlation) + correlation
            
            # Cholesky分解生成相关信号
            L = np.linalg.cholesky(corr_matrix)
            attack_signal = attack_signal @ L.T
        
        # 注入攻击
        for i, feat_idx in enumerate(target_features):
            original_values = attacked_data[start_idx:end_idx, feat_idx]
            
            # 攻击值：原始值 + 攻击信号
            attack_values = original_values * (1 + attack_signal[:, i])
            attacked_data[start_idx:end_idx, feat_idx] = attack_values
        
        # 攻击标签
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        
        # 攻击详情
        attack_info = {
            'type': 'multi_point',
            'start_idx': start_idx,
            'duration': duration,
            'target_features': target_features,
            'correlation': correlation,
            'description': f'多点攻击：特征{target_features}在[{start_idx}:{end_idx}]，相关性{correlation}'
        }
        
        return attacked_data, attack_labels, attack_info
    
    def inject_slow_drift_attack(self, start_idx, duration, target_feature, drift_rate=0.001):
        """
        注入缓慢漂移攻击
        
        参数：
        start_idx: 攻击开始位置
        duration: 攻击持续时间
        target_feature: 目标特征索引
        drift_rate: 每步漂移率
        """
        attacked_data = self.normal_data.copy()
        
        end_idx = min(start_idx + duration, self.n_samples)
        
        # 生成缓慢漂移信号（线性增加）
        drift_steps = end_idx - start_idx
        drift_signal = np.linspace(0, drift_rate * drift_steps, drift_steps)
        
        # 注入攻击
        original_values = attacked_data[start_idx:end_idx, target_feature]
        attack_values = original_values * (1 + drift_signal)
        attacked_data[start_idx:end_idx, target_feature] = attack_values
        
        # 攻击标签
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        
        # 攻击详情
        attack_info = {
            'type': 'slow_drift',
            'start_idx': start_idx,
            'duration': duration,
            'target_feature': target_feature,
            'drift_rate': drift_rate,
            'description': f'缓慢漂移攻击：特征{target_feature}在[{start_idx}:{end_idx}]，每步漂移{drift_rate*100}%'
        }
        
        return attacked_data, attack_labels, attack_info
    
    def inject_physical_constrained_attack(self, start_idx, duration, target_buses, attack_pattern='voltage_drop'):
        """
        注入物理约束攻击（保持电力系统物理规律）
        
        参数：
        start_idx: 攻击开始位置
        duration: 攻击持续时间
        target_buses: 目标总线列表
        attack_pattern: 攻击模式 ('voltage_drop', 'power_imbalance', 'phase_shift')
        """
        attacked_data = self.normal_data.copy()
        
        end_idx = min(start_idx + duration, self.n_samples)
        
        # 根据攻击模式生成物理一致的攻击
        if attack_pattern == 'voltage_drop':
            # 模拟电压下降，相关总线电压也相应变化
            for bus_idx in target_buses:
                # 电压特征索引（前14维是电压幅值）
                voltage_feature = bus_idx
                
                # 生成电压下降信号（平滑）
                original_voltages = attacked_data[start_idx:end_idx, voltage_feature]
                attack_factor = 0.9  # 下降到90%
                smooth_transition = np.linspace(1, attack_factor, end_idx - start_idx)
                
                attacked_data[start_idx:end_idx, voltage_feature] = original_voltages * smooth_transition
                
                # 相关总线也受影响（相邻总线）
                if bus_idx < 13:  # 如果不是最后一个总线
                    neighbor_feature = bus_idx + 1
                    neighbor_factor = 0.95  # 相邻总线影响较小
                    neighbor_signal = np.linspace(1, neighbor_factor, end_idx - start_idx)
                    original_neighbor = attacked_data[start_idx:end_idx, neighbor_feature]
                    attacked_data[start_idx:end_idx, neighbor_feature] = original_neighbor * neighbor_signal
        
        elif attack_pattern == 'power_imbalance':
            # 模拟功率不平衡攻击
            # 选择一个总线增加发电，另一个减少负载，保持系统大致平衡
            if len(target_buses) >= 2:
                # 第一个总线：增加发电
                gen_bus = target_buses[0]
                # 有功功率特征索引（28-41）
                gen_power_feature = 28 + gen_bus
                
                # 增加发电
                original_gen = attacked_data[start_idx:end_idx, gen_power_feature]
                gen_increase = np.linspace(0, 0.2, end_idx - start_idx)  # 增加20%
                attacked_data[start_idx:end_idx, gen_power_feature] = original_gen * (1 + gen_increase)
                
                # 第二个总线：减少负载
                load_bus = target_buses[1]
                load_power_feature = 28 + load_bus
                
                # 减少负载（注意：负载是负值，所以减少负载是增加）
                original_load = attacked_data[start_idx:end_idx, load_power_feature]
                load_decrease = np.linspace(0, -0.15, end_idx - start_idx)  # 减少15%
                attacked_data[start_idx:end_idx, load_power_feature] = original_load * (1 + load_decrease)
        
        # 攻击标签
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        
        # 攻击详情
        attack_info = {
            'type': 'physical_constrained',
            'start_idx': start_idx,
            'duration': duration,
            'target_buses': target_buses,
            'attack_pattern': attack_pattern,
            'description': f'物理约束攻击：总线{target_buses}，模式{attack_pattern}，区间[{start_idx}:{end_idx}]'
        }
        
        return attacked_data, attack_labels, attack_info
    
    def generate_attack_dataset(self, n_attacks=5, min_duration=50, max_duration=200, save_path="data/attack_data.pkl"):
        """
        生成包含多种攻击的数据集
        
        参数：
        n_attacks: 攻击事件数量
        min_duration: 最小攻击持续时间
        max_duration: 最大攻击持续时间
        save_path: 保存路径
        """
        print(f"开始生成攻击数据集，共{n_attacks}个攻击事件...")
        
        # 初始化
        all_attacked_data = []
        all_labels = []
        attack_infos = []
        
        # 生成多个攻击
        for attack_id in range(n_attacks):
            # 随机选择攻击类型
            attack_type = np.random.choice(list(self.attack_types.keys()))
            
            # 随机参数
            start_idx = np.random.randint(0, self.n_samples - max_duration)
            duration = np.random.randint(min_duration, max_duration)
            
            print(f"生成攻击 {attack_id+1}/{n_attacks}: {self.attack_types[attack_type]}")
            info = {}
            
            if attack_type == 'single_point':
                target_feature = np.random.randint(0, self.n_features)
                magnitude = np.random.uniform(0.1,0.2)  
                attacked_data, labels, info = self.inject_single_point_attack(
                    start_idx, duration, target_feature, magnitude
                )
                
            elif attack_type == 'multi_point':
                n_targets = np.random.randint(2, 5)  # 2-4个目标
                target_features = np.random.choice(self.n_features, n_targets, replace=False)
                correlation = np.random.uniform(0.5, 0.9)
                attacked_data, labels, info = self.inject_multi_point_attack(
                    start_idx, duration, target_features, correlation
                )
                
            elif attack_type == 'slow_drift':
                target_feature = np.random.randint(0, self.n_features)
                drift_rate = np.random.uniform(0.0005, 0.002)  # 0.05%-0.2%每步
                attacked_data, labels, info = self.inject_slow_drift_attack(
                    start_idx, duration, target_feature, drift_rate
                )
                
            elif attack_type == 'physical_constrained':
                n_buses = np.random.randint(2, 4)
                target_buses = np.random.choice(range(14), n_buses, replace=False)
                pattern = np.random.choice(['voltage_drop', 'power_imbalance'])
                attacked_data, labels, info = self.inject_physical_constrained_attack(
                    start_idx, duration, target_buses, pattern
                )
            
            info['attack_id'] = attack_id
            attack_infos.append(info)
            
            all_attacked_data.append(attacked_data)
            all_labels.append(labels)
        
        # 合并所有攻击（如果攻击有重叠，以最后一个为准）
        final_data = self.normal_data.copy()
        final_labels = np.zeros(self.n_samples, dtype=np.int32)
        
        for attacked_data, labels in zip(all_attacked_data, all_labels):
            # 找出攻击区域
            attack_indices = np.where(labels == 1)[0]
            if len(attack_indices) > 0:
                final_data[attack_indices] = attacked_data[attack_indices]
                final_labels[attack_indices] = 1
        
        # 保存数据
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'data': final_data,
                'labels': final_labels,
                'attack_infos': attack_infos,
                'normal_data': self.normal_data,
                'description': 'IEEE14系统FDIA攻击数据集'
            }, f)
        
        print(f"攻击数据集生成完成，保存到 {save_path}")
        print(f"数据形状: {final_data.shape}")
        print(f"攻击比例: {np.sum(final_labels)/len(final_labels)*100:.2f}%")
        
        return final_data, final_labels, attack_infos
    
    def generate_specific_attack_dataset(self, attack_type, n_attacks=3, min_duration=50, 
                                   max_duration=200, save_path_template="data/attack_{type}.pkl"):
        """
        生成特定类型的攻击数据集
        
        参数：
        attack_type: 攻击类型 ('single_point', 'multi_point', 'slow_drift', etc.)
        n_attacks: 攻击事件数量
        min_duration: 最小攻击持续时间
        max_duration: 最大攻击持续时间
        save_path_template: 保存路径模板，{type}会被替换为攻击类型
        """
        if attack_type not in self.attack_types:
            raise ValueError(f"无效的攻击类型，可选: {list(self.attack_types.keys())}")
        
        print(f"开始生成{self.attack_types[attack_type]}数据集，共{n_attacks}个攻击事件...")
        
        # 生成攻击数据集（逻辑与generate_attack_dataset类似，但只生成指定类型）
        all_attacked_data = []
        all_labels = []
        attack_infos = []
        
        for attack_id in range(n_attacks):
            # 随机参数
            start_idx = np.random.randint(0, self.n_samples - max_duration)
            duration = np.random.randint(min_duration, max_duration)
            
            print(f"生成攻击 {attack_id+1}/{n_attacks}: {self.attack_types[attack_type]}")
            
            # 根据攻击类型调用相应方法
            if attack_type == 'single_point':
                target_feature = np.random.randint(0, self.n_features)
                magnitude = np.random.uniform(0.1,0.2)
                attacked_data, labels, info = self.inject_single_point_attack(
                    start_idx, duration, target_feature, magnitude
                )
                
            elif attack_type == 'multi_point':
                n_targets = np.random.randint(2, 5)
                target_features = np.random.choice(self.n_features, n_targets, replace=False)
                correlation = np.random.uniform(0.5, 0.9)
                attacked_data, labels, info = self.inject_multi_point_attack(
                    start_idx, duration, target_features, correlation
                )
                
            elif attack_type == 'slow_drift':
                target_feature = np.random.randint(0, self.n_features)
                drift_rate = np.random.uniform(0.0005, 0.002)
                attacked_data, labels, info = self.inject_slow_drift_attack(
                    start_idx, duration, target_feature, drift_rate
                )
                
            elif attack_type == 'physical_constrained':
                n_buses = np.random.randint(2, 4)
                target_buses = np.random.choice(range(14), n_buses, replace=False)
                pattern = np.random.choice(['voltage_drop', 'power_imbalance'])
                attacked_data, labels, info = self.inject_physical_constrained_attack(
                    start_idx, duration, target_buses, pattern
                )
            
            info['attack_id'] = attack_id
            attack_infos.append(info)
            all_attacked_data.append(attacked_data)
            all_labels.append(labels)
        
        # 合并所有攻击
        final_data = self.normal_data.copy()
        final_labels = np.zeros(self.n_samples, dtype=np.int32)
        
        for attacked_data, labels in zip(all_attacked_data, all_labels):
            attack_indices = np.where(labels == 1)[0]
            if len(attack_indices) > 0:
                final_data[attack_indices] = attacked_data[attack_indices]
                final_labels[attack_indices] = 1
        
        # 保存数据
        save_path = save_path_template.format(type=attack_type)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump({
                'data': final_data,
                'labels': final_labels,
                'attack_infos': attack_infos,
                'normal_data': self.normal_data,
                'attack_type': attack_type,
                'description': f'IEEE14系统{self.attack_types[attack_type]}数据集'
            }, f)
        
        print(f"{self.attack_types[attack_type]}数据集生成完成，保存到 {save_path}")
        print(f"数据形状: {final_data.shape}")
        print(f"攻击比例: {np.sum(final_labels)/len(final_labels)*100:.2f}%")
        
        return final_data, final_labels, attack_infos

    def visualize_attacks(self, attacked_data, labels, attack_infos, save_path=None):
        """
        可视化攻击效果
        
        参数：
        attacked_data: 攻击数据
        labels: 攻击标签
        attack_infos: 攻击信息列表
        save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 选择几个代表性特征进行可视化
        sample_features = [0, 14, 28, 42]  # 总线0的电压、相角、有功、无功
        
        # 1. 原始数据 vs 攻击数据
        time_indices = range(len(attacked_data))
        
        for i, feat_idx in enumerate(sample_features):
            axes[0].plot(time_indices, self.normal_data[:, feat_idx], 
                        alpha=0.7, label=f'特征{feat_idx}-正常')
            axes[0].plot(time_indices, attacked_data[:, feat_idx], 
                        alpha=0.7, linestyle='--', label=f'特征{feat_idx}-攻击')
        
        axes[0].set_title('正常数据 vs 攻击数据')
        axes[0].set_ylabel('测量值')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=2)
        
        # 2. 攻击标签
        axes[1].fill_between(time_indices, 0, labels, alpha=0.5, color='red')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_title('攻击标签 (1=攻击, 0=正常)')
        axes[1].set_ylabel('攻击状态')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 残差（攻击数据 - 正常数据）
        residuals = attacked_data - self.normal_data
        residual_norm = np.linalg.norm(residuals, axis=1)
        
        axes[2].plot(time_indices, residual_norm, color='green')
        axes[2].fill_between(time_indices, 0, residual_norm, alpha=0.3, color='green')
        axes[2].set_title('攻击残差范数')
        axes[2].set_ylabel('残差范数')
        axes[2].set_xlabel('时间步')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"攻击可视化保存到 {save_path}")
        
        plt.show()
        
        # 打印攻击统计信息
        print("\n攻击统计信息:")
        print(f"总时间步数: {len(labels)}")
        print(f"攻击时间步数: {np.sum(labels)}")
        print(f"攻击比例: {np.sum(labels)/len(labels)*100:.2f}%")
        
        for info in attack_infos:
            # 安全地获取描述信息
            attack_id = info.get('attack_id', '未知')
            description = info.get('description', '无描述')
            attack_type = info.get('type', '未知类型')
            
            print(f"\n攻击 {attack_id}: 类型={attack_type}, {description}")


if __name__ == "__main__":
    # 配置参数
    config = {
        'sampling_rate': 5,      # 1Hz采样
        'total_hours': 3,         # 3小时数据
        'window_size': 10,        # 10个时间步的窗口
        'step': 5,               # 滑动步长为5
        'n_attacks': 200,         # 攻击事件数量：生成200个攻击实例
        'min_duration': 90,      # 最小攻击持续时间：每个攻击至少持续80个时间步
        'max_duration': 110,     # 最大攻击持续时间：每个攻击最多持续120个时间步
        'train_ratio': 0.7,      # 70%训练集
        'val_ratio': 0.15        # 15%验证集，15%测试集
    }
    
    # 创建并运行流水线
    pipeline = FDIA_DataPipeline(config)
    
    # 运行完整流水线（第一次运行或需要重新生成数据时）single_point', 'multi_point', 'slow_drift','physical_constrained'
    pipeline.run_full_pipeline(attack_type="single_point",regenerate=True)
    
    # 或者加载已处理的数据（如果已经运行过）
    pipeline.load_processed_data("processed_data")
    
    # # 获取数据加载器
    # train_loader, val_loader, test_loader = pipeline.get_data_loaders(batch_size=32)
    
    # # 打印摘要信息
    # pipeline.summarize()
    
    # # 测试一个批次
    # print("\n测试一个批次的数据:")
    # for batch_X, batch_y in train_loader:
    #     print(f"批次形状: X={batch_X.shape}, y={batch_y.shape}")
    #     print(f"批次标签分布: 正常={torch.sum(batch_y==0).item()}, 攻击={torch.sum(batch_y==1).item()}")
    #     break
