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
pd.options.mode.chained_assignment = None  # 关闭链式赋值警告
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
        self.base_loads = {}            # 初始化base_loads
        self.load_history = []          # 记录每个时间步的负荷值 (n_samples, n_loads)
        self.convergence_history = []   # 记录收敛标志 (bool)
        self.load_random_std = 0.05
        
    def create_ieee14_network(self):
        """创建IEEE14系统，使用标准测试系统参数"""
        self.net = nw.case14() # 使用默认网络

        if hasattr(self.net, 'shunt') and len(self.net.shunt) > 0:
            self.net.shunt['q_mvar'] = self.net.shunt['q_mvar'].abs()

        self.base_loads = {}
        for idx in self.net.load.index:
            self.base_loads[idx] = {
                'p_mw': self.net.load.at[idx, 'p_mw'],
                'q_mvar': self.net.load.at[idx, 'q_mvar']
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
            random_factor = 1 + np.random.normal(0, 0.02)  # ±2%随机波动
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
        
        # 简单日负荷曲线
        # 缩放到 0.8~1.2，保持形状相似
        original = np.array([0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 
                     0.9, 1.0, 0.95, 0.9, 0.85, 0.8, 0.8, 0.85,
                     0.9, 0.95, 1.0, 1.0, 0.95, 0.9, 0.8, 0.7])
        # 映射到 [0.8, 1.2]
        min_orig = original.min()  # 0.5
        max_orig = original.max()  # 1.0
        day_profile = 0.8 + (original - min_orig) / (max_orig - min_orig) * (1.2 - 0.8)
        
        # 计算时间相关因子
        # 计算连续小时（例如 2.5 表示 2:30）
        hour_cont = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

        # 创建小时点数组 0..23
        hours = np.arange(24)

        # 对 day_profile 进行线性插值，得到连续负荷因子
        load_factor = np.interp(hour_cont, hours, day_profile)
        
        # 2. 更新所有负载
        for load_idx in self.net.load.index:
            if load_idx in self.base_loads:
                base_p = self.base_loads[load_idx]['p_mw']
                base_q = self.base_loads[load_idx]['q_mvar']
                
                # 添加随机波动
                # random_factor = 1 + np.random.normal(0, 0.05)  # ±2%波动
                random_std = getattr(self, 'load_random_std', 0.02)  # 默认 2%
                random_factor = 1 + np.random.normal(0, random_std)
                current_factor = load_factor * random_factor
                
                self.net.load.at[load_idx, 'p_mw'] = base_p * current_factor
                self.net.load.at[load_idx, 'q_mvar'] = base_q * current_factor
        
        # 3. 执行潮流计算
        try:
            pp.runpp(self.net)
            # 记录负荷值
            load_p = self.net.load['p_mw'].values.copy()
            self.load_history.append(load_p)
            self.convergence_history.append(True)
            return self.extract_measurements()
        except Exception as e:
            print(f"潮流计算失败: {e}")
            # 记录失败状态（保守估计时负荷值可能无变化，这里可重复上一次的值或基准值）
            if self.load_history:
                self.load_history.append(self.load_history[-1])
            else:
                # 若第一次就失败，从基准负荷构建
                base_p = [self.base_loads[load_idx]['p_mw'] for load_idx in sorted(self.net.load.index)]
                self.load_history.append(np.array(base_p))
            self.convergence_history.append(False)
            return self._generate_conservative_estimate(timestamp)
    
    def extract_measurements(self):
        """从网络提取测量值"""
        if self.net is None:
            print("警告: 网络未初始化")
            return np.zeros(self.measurement_dim, dtype=np.float32)
        
        measurements = np.zeros(self.measurement_dim, dtype=np.float32)
        
        # 添加测量噪声
        noise_std = {
            'v': 0.004,      # 电压幅值标准差 0.004 p.u.
            'va': 0.1,       # 相角度
            'p': 0.01,       # 有功功率 MW
            'q': 0.01        # 无功功率 MVar
        }

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
        
        # 电压幅值噪声
        measurements[0:14] += np.random.normal(0, noise_std['v'], 14)
        # 相角噪声
        measurements[14:28] += np.random.normal(0, noise_std['va'], 14)
        # 有功噪声
        measurements[28:42] += np.random.normal(0, noise_std['p'], 14)
        # 无功噪声
        measurements[42:56] += np.random.normal(0, noise_std['q'], 14)
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
    
    def generate_normal_data(self, save_path=None, workers=None):
        """
        生成正常状态下的数据
        
        参数:
        save_path: 数据保存路径
        workers
        
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
    
    def get_base_measurements(self):
        """运行一次基准潮流，返回测量值"""
        if self.net is None:
            self.create_ieee14_network()
        # 保存当前负荷状态
        orig_loads = self.net.load[['p_mw', 'q_mvar']].copy()
        # 恢复基准负荷（self.base_loads 中已保存）
        for load_idx in self.net.load.index:
            if load_idx in self.base_loads:
                self.net.load.at[load_idx, 'p_mw'] = self.base_loads[load_idx]['p_mw']
                self.net.load.at[load_idx, 'q_mvar'] = self.base_loads[load_idx]['q_mvar']
        try:
            pp.runpp(self.net)
            base_meas = self.extract_measurements()
        except Exception as e:
            print(f"基准潮流计算失败: {e}")
            base_meas = np.zeros(self.measurement_dim)
        finally:
            # 恢复原来的负荷
            self.net.load[['p_mw', 'q_mvar']] = orig_loads
        return base_meas
    
    

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
        # 添加 IEEE14 节点系统的母线连接关系（母线索引从0开始）
        self.connections = [
            (0,1), (0,4), (1,2), (1,3), (1,4), (2,3), (3,4), 
            (3,6), (3,8), (4,5), (5,10), (5,11), (5,12), 
            (6,7), (6,8), (8,9), (8,13), (9,10), (11,12), (12,13)
        ]
        self.gen_buses = [0, 1, 2, 5, 7]  # IEEE14 的发电机母线（包括松弛节点）
    
    def inject_single_point_attack(self, start_idx, duration, target_feature=None, strength=3.0, prefer_power=True):
        """
        单点攻击：只篡改一个测量点
        参数：
            target_feature: 指定攻击特征（若为None则随机选择）
            prefer_power: 是否优先选择功率量测（特征28-55）
        """
        attacked_data = self.normal_data.copy()
        end_idx = min(start_idx + duration, self.n_samples)

        # 定义不适合攻击的特征（松弛节点和PV节点的电压幅值）
        # 母线索引: 0(松弛),1,2,5,7(PV) -> 电压特征索引 0,1,2,5,7
        excluded_voltage = [0, 1, 2, 5, 7]  # 松弛节点+PV节点电压幅值
        # 可选：也可排除这些母线的相角特征（14+索引），但相角影响较小，可保留
        excluded_features = excluded_voltage  # 目前仅排除电压幅值

        if target_feature is None:
            # 获取所有有效特征（排除 excluded_features）
            all_features = list(range(self.n_features))
            valid_features = [f for f in all_features if f not in excluded_features]

            if prefer_power:
                # 功率量测特征索引范围 28-55
                power_features = [f for f in valid_features if 28 <= f <= 55]
                voltage_features = [f for f in valid_features if f < 14]  # PQ节点的电压幅值
                other_features = [f for f in valid_features if f not in power_features + voltage_features]

                # 设置采样权重：功率量测权重高，电压量测权重低，其他量测权重中等
                features_pool = []
                weights = []
                if power_features:
                    features_pool.extend(power_features)
                    weights.extend([3] * len(power_features))  # 功率量测权重3
                if voltage_features:
                    features_pool.extend(voltage_features)
                    weights.extend([1] * len(voltage_features))  # PQ节点电压权重1
                if other_features:
                    features_pool.extend(other_features)
                    weights.extend([2] * len(other_features))    # 其他（如相角）权重2

                # 归一化权重
                weights = np.array(weights) / np.sum(weights)
                target_feature = np.random.choice(features_pool, p=weights)
            else:
                # 等概率随机选择
                target_feature = np.random.choice(valid_features)

        # 计算攻击偏差（使用特征标准差，现在因噪声存在均非零）
        feat_std = np.std(self.normal_data[:, target_feature])
        # 为防止意外（如特征全零），仍保留备用基准
        if feat_std < 1e-6:
            if target_feature < 14:
                base = 0.01
            elif target_feature < 28:
                base = 0.5
            else:
                base = 0.1
            feat_std = base

        direction = np.random.choice([-1, 1])
        attack_bias = direction * strength * feat_std
        attacked_data[start_idx:end_idx, target_feature] += attack_bias

        # 标签与信息
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        attack_info = {
            'type': 'single_point',
            'start_idx': start_idx,
            'duration': duration,
            'target_feature': int(target_feature),
            'strength': strength,
            'affected_features': [int(target_feature)],
            'description': f'单点攻击：特征{target_feature}，强度{strength}σ，方向{direction}'
        }
        return attacked_data, attack_labels, attack_info
    
    def inject_multi_point_attack(self, start_idx, duration, target_features, correlation=0.8, strength=3.0):
        """
        注入多点协同攻击（基于标准差）
        
        参数：
            start_idx: 攻击开始位置
            duration: 攻击持续时间
            target_features: 目标特征列表
            correlation: 特征间相关性（0-1）
            strength: 攻击强度（相对于各特征标准差的倍数）
        """
        attacked_data = self.normal_data.copy()
        end_idx = min(start_idx + duration, self.n_samples)
        n_targets = len(target_features)
        
        # 生成相关攻击信号
        base_signal = np.random.normal(0, 1, size=(end_idx - start_idx, n_targets))
        corr_matrix = np.eye(n_targets) * (1 - correlation) + correlation
        L = np.linalg.cholesky(corr_matrix)
        correlated_signal = base_signal @ L.T  # 形状 (T, n_targets)
        
        # 注入攻击
        for i, feat_idx in enumerate(target_features):
            feat_std = np.std(self.normal_data[:, feat_idx])
            # 攻击偏差 = 相关信号 * strength * feat_std
            attack_bias = correlated_signal[:, i] * strength * feat_std
            attacked_data[start_idx:end_idx, feat_idx] += attack_bias
        
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
            'strength': strength,   # 新增字段
            'affected_features': target_features,
            'description': f'多点攻击：特征{target_features}，强度{strength}σ，相关性{correlation}'
        }

        return attacked_data, attack_labels, attack_info
    
    def inject_slow_drift_attack(self, start_idx, duration, target_feature, strength=5.0):
        """
        缓慢漂移攻击：最终偏差达到 strength * std
        """
        attacked_data = self.normal_data.copy()
        end_idx = min(start_idx + duration, self.n_samples)
        feat_std = np.std(self.normal_data[:, target_feature])
        
        # 漂移方向随机
        direction = np.random.choice([-1, 1])
        # 漂移信号：从0线性增加到 direction * strength * feat_std
        drift_signal = np.linspace(0, direction * strength * feat_std, end_idx - start_idx)
        
        attacked_data[start_idx:end_idx, target_feature] += drift_signal
        
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        
        attack_info = {
            'type': 'slow_drift',
            'start_idx': start_idx,
            'duration': duration,
            'target_feature': target_feature,
            'strength': strength,
            'affected_features': [target_feature],
            'description': f'缓慢漂移攻击：特征{target_feature}，强度{strength}σ'
        }


        return attacked_data, attack_labels, attack_info
    
    def inject_physical_constrained_attack(self, start_idx, duration, target_buses, attack_pattern='power_imbalance', strength=3.0):
        attacked_data = self.normal_data.copy()
        end_idx = min(start_idx + duration, self.n_samples)
        affected = []  # 初始化受影响特征列表

        if attack_pattern == 'voltage_drop':
            for bus_idx in target_buses:
                # 电压特征索引
                v_feat = bus_idx
                # 无功特征索引
                q_feat = 42 + bus_idx

                v_std = np.std(self.normal_data[:, v_feat])
                # 生成电压下降曲线
                v_drop = np.linspace(0, strength * v_std, end_idx - start_idx)
                attacked_data[start_idx:end_idx, v_feat] -= v_drop
                affected.append(v_feat)

                # 计算电压下降的比例（相对于攻击开始时刻的电压值）
                base_v = self.normal_data[start_idx, v_feat]
                if base_v != 0:
                    drop_ratio = v_drop / base_v
                    # 按相同比例减小无功（假设负荷功率因数不变）
                    base_q = self.normal_data[start_idx, q_feat]
                    q_drop = drop_ratio * base_q
                    attacked_data[start_idx:end_idx, q_feat] -= q_drop
                    affected.append(q_feat)
                else:
                    # 如果基准电压为零（几乎不可能），则跳过无功调整
                    pass

                # 相邻母线（假设为 bus_idx+1，可根据实际情况修改）
                if bus_idx < 13:
                    neighbor_v = bus_idx + 1
                    neighbor_q = 42 + (bus_idx + 1)

                    neighbor_v_std = np.std(self.normal_data[:, neighbor_v])
                    neighbor_v_drop = np.linspace(0, strength * 0.5 * neighbor_v_std, end_idx - start_idx)
                    attacked_data[start_idx:end_idx, neighbor_v] -= neighbor_v_drop
                    affected.append(neighbor_v)

                    # 相邻母线无功按相同比例调整
                    base_neighbor_v = self.normal_data[start_idx, neighbor_v]
                    if base_neighbor_v != 0:
                        neighbor_drop_ratio = neighbor_v_drop / base_neighbor_v
                        base_neighbor_q = self.normal_data[start_idx, neighbor_q]
                        neighbor_q_drop = neighbor_drop_ratio * base_neighbor_q
                        attacked_data[start_idx:end_idx, neighbor_q] -= neighbor_q_drop
                        affected.append(neighbor_q)

        elif attack_pattern == 'power_imbalance':
            if len(target_buses) >= 2:
                gen_bus = target_buses[0]
                load_bus = target_buses[1]

                # 发电母线有功增加
                gen_feature = 28 + gen_bus
                gen_std = np.std(self.normal_data[:, gen_feature])
                gen_increase = np.linspace(0, strength * gen_std, end_idx - start_idx)  # 形状 (duration,)
                attacked_data[start_idx:end_idx, gen_feature] += gen_increase
                affected.append(gen_feature)

                # ----- 改进点：将部分增加的功率分配到相邻负荷节点 -----
                # 获取与发电母线直接相连的母线
                neighbors = []
                for (i, j) in self.connections:  # 需要 self.connections 在类初始化时定义
                    if i == gen_bus:
                        neighbors.append(j)
                    elif j == gen_bus:
                        neighbors.append(i)
                neighbors = list(set(neighbors))  # 去重

                # 筛选出负荷节点（正常运行时平均有功为正）
                load_neighbors = []
                for nb in neighbors:
                    feat_p = 28 + nb
                    if np.mean(self.normal_data[:, feat_p]) > 0:  # 平均有功为正，认为是负荷
                        load_neighbors.append(nb)

                if load_neighbors:
                    # 计算每个相邻负荷节点的分配权重（基于其原始有功大小）
                    base_powers = [np.mean(self.normal_data[:, 28 + nb]) for nb in load_neighbors]
                    total_base = sum(base_powers)
                    if total_base > 0:
                        weights = [p / total_base for p in base_powers]
                    else:
                        weights = [1.0 / len(load_neighbors)] * len(load_neighbors)

                    # 对每个时间步，将 gen_increase[t] 分配到各相邻负荷节点
                    for t_idx, t in enumerate(range(start_idx, end_idx)):
                        inc_t = gen_increase[t_idx]  # 当前时间步的发电增加量
                        for w, nb in zip(weights, load_neighbors):
                            nb_feat = 28 + nb
                            # 增加该负荷节点的有功测量值
                            attacked_data[t, nb_feat] += w * inc_t
                            if nb_feat not in affected:
                                affected.append(nb_feat)

                # 负载总线：减少负载（需考虑负载的符号）
                load_feature = 28 + load_bus
                load_std = np.std(self.normal_data[:, load_feature])
                load_decrease = np.linspace(0, strength * load_std, end_idx - start_idx)
                # 判断负载的典型符号（正为消耗，负为注入）
                load_sign = np.sign(np.mean(self.normal_data[:, load_feature]))
                if load_sign > 0:  # 负载为正，减少负载即减去偏差
                    attacked_data[start_idx:end_idx, load_feature] -= load_decrease
                else:               # 负载为负，减少负载即加上偏差（向零靠近）
                    attacked_data[start_idx:end_idx, load_feature] += load_decrease
                affected.append(load_feature)

        # 攻击标签
        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1

        # 攻击详情（直接包含 affected_features）
        attack_info = {
            'type': 'physical_constrained',
            'start_idx': start_idx,
            'duration': duration,
            'target_buses': target_buses,
            'attack_pattern': attack_pattern,
            'strength': strength,
            'description': f'物理约束攻击：总线{target_buses}，模式{attack_pattern}，强度{strength}σ',
            'affected_features': list(set(affected))  # 去重，避免重复特征
        }
        
        return attacked_data, attack_labels, attack_info
    
    def generate_attack_dataset(self, n_attacks=5, min_duration=50, max_duration=200, strength_range=(3.0, 6.0), significance_threshold=3.0,
                            max_retries=3, save_path="data/attack_data.pkl"):
        """
        生成包含多种攻击的数据集
        
        参数：
        n_attacks: 攻击事件数量
        min_duration: 最小攻击持续时间
        max_duration: 最大攻击持续时间
        strength_range: 攻击强度范围（元组）
        significance_threshold: 显著性阈值（倍数）
        max_retries: 每个攻击的最大重试次数
        save_path: 保存路径
        """
        print(f"开始生成攻击数据集，共{n_attacks}个攻击事件...")
        
        # 初始化
        all_attacked_data = []
        all_labels = []
        attack_infos = []
        
        # 生成多个攻击
        for attack_id in range(n_attacks):
            retries = 0
            success = False
            while retries < max_retries and not success:
                # 随机选择攻击类型
                implemented_attacks = ['single_point', 'multi_point', 'slow_drift', 'physical_constrained']
                attack_type = np.random.choice(implemented_attacks)
                
                # 随机参数
                start_idx = np.random.randint(0, self.n_samples - max_duration)
                duration = np.random.randint(min_duration, max_duration)
                strength = np.random.uniform(*strength_range)
            
                info = {}
            
                if attack_type == 'single_point':
                    target_feature = np.random.randint(0, self.n_features)  
                    attacked_data, labels, info = self.inject_single_point_attack(
                        start_idx, duration, target_feature, strength
                    )
                    
                elif attack_type == 'multi_point':
                    n_targets = np.random.randint(2, 5)
                    target_features = np.random.choice(self.n_features, n_targets, replace=False)
                    correlation = np.random.uniform(0.5, 0.9)
                    attacked_data, labels, info = self.inject_multi_point_attack(
                        start_idx, duration, target_features, correlation, strength
                    )
                    
                elif attack_type == 'slow_drift':
                    target_feature = np.random.randint(0, self.n_features)
                    # 使用 strength，不再使用 drift_rate
                    attacked_data, labels, info = self.inject_slow_drift_attack(
                        start_idx, duration, target_feature, strength
                    )
                elif attack_type == 'physical_constrained':
                    n_buses = np.random.randint(2, 4)
                    target_buses = np.random.choice(range(14), n_buses, replace=False)
                    pattern = np.random.choice(['voltage_drop', 'power_imbalance'])
                    attacked_data, labels, info = self.inject_physical_constrained_attack(
                        start_idx, duration, target_buses, pattern, strength
                    )
                # 验证显著性
                significant, details = self.validate_attack_significance(
                    attacked_data, info, threshold=significance_threshold
                )
                if significant:
                    success = True
                    info['significance_details'] = details
                else:
                    print(f"攻击显著性不足 (ratio={details[list(details.keys())[0]]['ratio']:.2f})，重试 {retries+1}/{max_retries}")
                    retries += 1
            if not success:
                print(f"警告：攻击 {attack_id+1} 在 {max_retries} 次尝试后仍不显著，使用最后一次结果")

                     
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
        
        # 物理约束剪裁
        final_data[:, :14] = np.clip(final_data[:, :14], 0.85, 1.15)
        
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
                                   max_duration=200, strength_range=(6.0, 10.0), save_path_template="data/attack_{type}.pkl"):
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
            strength = np.random.uniform(*strength_range)
            
            
            # 根据攻击类型调用相应方法
            if attack_type == 'single_point':
                target_feature = np.random.randint(0, self.n_features)
                attacked_data, labels, info = self.inject_single_point_attack(
                    start_idx, duration, target_feature, strength
                )
                
            elif attack_type == 'multi_point':
                n_targets = np.random.randint(2, 5)
                target_features = np.random.choice(self.n_features, n_targets, replace=False)
                correlation = np.random.uniform(0.5, 0.9)
                attacked_data, labels, info = self.inject_multi_point_attack(
                    start_idx, duration, target_features, correlation,strength
                )
                
            elif attack_type == 'slow_drift':
                target_feature = np.random.randint(0, self.n_features)
                attacked_data, labels, info = self.inject_slow_drift_attack(
                    start_idx, duration, target_feature, strength
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
    
    def validate_attack_significance(self, attacked_data, attack_info, threshold=3.0):
        start = attack_info['start_idx']
        end = min(start + attack_info['duration'], self.n_samples)
        
        # 1. 确定要检查的特征
        if 'affected_features' in attack_info:
            check_features = attack_info['affected_features']
        else:
            # 向后兼容
            if 'target_feature' in attack_info:
                check_features = [attack_info['target_feature']]
            elif 'target_features' in attack_info:
                check_features = attack_info['target_features']
            elif 'target_buses' in attack_info:
                check_features = attack_info['target_buses']  # 假设是电压特征
            else:
                check_features = []
        
        # 2. 根据攻击类型选择度量方式
        attack_type = attack_info.get('type', '')
        use_max = attack_type in ['slow_drift', 'physical_constrained']
        
        details = {}
        overall_significant = False  # 初始为False，只要有一个显著就设为True
        
        for feat in check_features:
            normal_seg = self.normal_data[start:end, feat]
            attack_seg = attacked_data[start:end, feat]
            deviation = attack_seg - normal_seg
            feat_std = np.std(self.normal_data[:, feat])
            
            if feat_std == 0:
                ratio = np.inf if np.any(deviation != 0) else 0
            else:
                if use_max:
                    ratio = np.max(np.abs(deviation)) / feat_std
                else:
                    ratio = abs(np.mean(deviation)) / feat_std
            
            details[feat] = {
                'ratio': ratio,
                'significant': ratio >= threshold
            }
            if ratio >= threshold:
                overall_significant = True
                   
        return overall_significant, details

    def visualize_attack_impact(self, attacked_data, attack_info, save_path=None, 
                            pad_before=50, pad_after=50, figsize=None):
        """
        改进版：直观展示攻击影响
        参数:
            attacked_data: 攻击数据
            attack_info: 攻击信息字典
            save_path: 保存路径（支持 .png, .pdf 等）
            pad_before: 攻击前额外显示的时间步数
            pad_after: 攻击后额外显示的时间步数
            figsize: 图形尺寸，默认根据特征数量自动调整
        """
        start = attack_info['start_idx']
        duration = attack_info['duration']
        end = start + duration
        
        # 确定受影响的特征
        if 'affected_features' in attack_info:
            features = attack_info['affected_features']
        elif 'target_feature' in attack_info:
            features = [attack_info['target_feature']]
        elif 'target_features' in attack_info:
            features = attack_info['target_features']
        elif 'target_buses' in attack_info:
            # 假设电压特征（0-13）受影响，可根据需要扩展
            features = list(attack_info['target_buses'])
        else:
            features = list(range(min(5, self.n_features)))  # 默认前5个
        
        # 计算显示区间
        plot_start = max(0, start - pad_before)
        plot_end = min(self.n_samples, end + pad_after)
        time = np.arange(plot_start, plot_end)
        
        # 创建子图
        n_features = len(features)
        if n_features == 0:
            print("警告：没有指定受影响特征，无法绘图")
            return
        if figsize is None:
            figsize = (12, 4 * n_features)
        
        fig, axes = plt.subplots(n_features, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for idx, feat in enumerate(features):
            ax = axes[idx]
            
            # 提取数据
            normal_seg = self.normal_data[plot_start:plot_end, feat]
            attack_seg = attacked_data[plot_start:plot_end, feat]
            
            # 绘制正常和攻击曲线
            ax.plot(time, normal_seg, label='正常', color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.plot(time, attack_seg, label='攻击', color='red', linestyle='-', linewidth=1.5)
            # 标记攻击区间（半透明矩形）
            ax.axvspan(start, end, alpha=0.2, color='red', label='攻击时段')
            
            # 标注显著性（如果可用）
            if 'significance_details' in attack_info and feat in attack_info['significance_details']:
                ratio = attack_info['significance_details'][feat]['ratio']
                # 在攻击区间内找到偏差最大的位置
                attack_slice = attack_seg[(time >= start) & (time < end)]
                normal_slice = normal_seg[(time >= start) & (time < end)]
                if len(attack_slice) > 0:
                    # 计算最大偏差点
                    max_dev_idx = np.argmax(np.abs(attack_slice - normal_slice))
                    max_time = time[(time >= start) & (time < end)][max_dev_idx]
                    max_val = attack_slice[max_dev_idx]
                    # 在图上添加箭头和文本
                    ax.annotate(f'{ratio:.1f}σ', 
                        xy=(max_time, max_val),
                        xytext=(0, 10),                 # 向右偏移0点，向上偏移10点
                        textcoords='offset points',     # 使用点坐标偏移
                        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                        fontsize=12, color='black',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9),
                        ha='center', va='bottom')        # 水平居中，垂直底部对齐
            
            # 装饰
            ax.set_xlabel('时间步', fontsize=11)
            ax.set_ylabel(f'特征 {feat} 值', fontsize=11)
            ax.set_title(f'特征 {feat} 受攻击影响', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 设置纵轴范围留白
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min - 0.05*(y_max-y_min), y_max + 0.05*(y_max-y_min))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"攻击影响图已保存到 {save_path}")
        plt.show()

    def generate_mixed_dataset_with_split(self, split_ratio=0.85,
                                      known_attack_types=None,
                                      unknown_attack_types=None,
                                      total_attack_ratio=0.5,
                                      min_duration=50, max_duration=200,
                                      strength_range=(3.0, 8.0),
                                      save_path=None):
        if known_attack_types is None:
            known_attack_types = ['single_point', 'multi_point', 'slow_drift']
        if unknown_attack_types is None:
            unknown_attack_types = ['physical_constrained']
        all_attack_types = known_attack_types + unknown_attack_types

        # 计算分段点
        split_idx = int(self.n_samples * split_ratio)
        train_len = split_idx
        test_len = self.n_samples - split_idx

        # 计算各段目标攻击样本数
        train_target = int(train_len * total_attack_ratio)
        test_target = int(test_len * total_attack_ratio)
        # 测试段中每种攻击类型的目标样本数（平均分配）
        type_target = test_target // len(all_attack_types)

        # 初始化最终数据
        final_data = self.normal_data.copy()
        final_labels = np.zeros(self.n_samples, dtype=np.int32)
        all_infos = []

        # --- 生成训练验证段攻击（仅已知类型）---
        train_attacks_needed = train_target
        train_attacks_generated = 0
        train_attack_infos = []
        while train_attacks_generated < train_attacks_needed:
            # 随机选择一种已知攻击类型
            atype = np.random.choice(known_attack_types)
            # 随机参数
            start = np.random.randint(0, train_len - max_duration)
            duration = np.random.randint(min_duration, max_duration)
            strength = np.random.uniform(*strength_range)

            if atype == 'single_point':
                target_feature = np.random.randint(0, self.n_features)
                attacked, labels, info = self.inject_single_point_attack(
                    start, duration, target_feature, strength)
            elif atype == 'multi_point':
                n_targets = np.random.randint(2, 5)
                target_features = np.random.choice(self.n_features, n_targets, replace=False)
                corr = np.random.uniform(0.5, 0.9)
                attacked, labels, info = self.inject_multi_point_attack(
                    start, duration, target_features, corr, strength)
            elif atype == 'slow_drift':
                target_feature = np.random.randint(0, self.n_features)
                attacked, labels, info = self.inject_slow_drift_attack(
                    start, duration, target_feature, strength)
            else:
                continue

            # 验证显著性（可选）
            significant, _ = self.validate_attack_significance(attacked, info)
            if not significant:
                continue

            train_attack_infos.append(info)
            train_attacks_generated += duration

        # 合并训练验证段攻击
        for info in train_attack_infos:
            s = info['start_idx']
            e = s + info['duration']
            final_data[s:e] = attacked[s:e]  # 注意：attacked 是最后一次攻击的局部数组，这里需修正
            final_labels[s:e] = 1
            all_infos.append(info)
        # 注意：上述合并需要每次攻击后保存局部数组，更好的方式是直接在此循环内更新 final_data
        # 我们将在后续优化，此处仅为示意。

        # --- 生成测试段攻击（所有类型，均衡分布）---
        # 为每种攻击类型生成攻击事件，直到累计样本数达到 type_target
        for atype in all_attack_types:
            generated = 0
            while generated < type_target:
                start = np.random.randint(split_idx, self.n_samples - max_duration)
                duration = np.random.randint(min_duration, max_duration)
                strength = np.random.uniform(*strength_range)

                if atype == 'single_point':
                    target_feature = np.random.randint(0, self.n_features)
                    attacked, labels, info = self.inject_single_point_attack(
                        start, duration, target_feature, strength)
                elif atype == 'multi_point':
                    n_targets = np.random.randint(2, 5)
                    target_features = np.random.choice(self.n_features, n_targets, replace=False)
                    corr = np.random.uniform(0.5, 0.9)
                    attacked, labels, info = self.inject_multi_point_attack(
                        start, duration, target_features, corr, strength)
                elif atype == 'slow_drift':
                    target_feature = np.random.randint(0, self.n_features)
                    attacked, labels, info = self.inject_slow_drift_attack(
                        start, duration, target_feature, strength)
                elif atype == 'physical_constrained':
                    n_buses = np.random.randint(2, 4)
                    target_buses = np.random.choice(range(14), n_buses, replace=False)
                    pattern = np.random.choice(['voltage_drop', 'power_imbalance'])
                    attacked, labels, info = self.inject_physical_constrained_attack(
                        start, duration, target_buses, pattern, strength)

                # 验证显著性
                significant, _ = self.validate_attack_significance(attacked, info)
                if not significant:
                    continue

                # 更新 final_data 和 final_labels
                final_data[start:start+duration] = attacked[start:start+duration]
                final_labels[start:start+duration] = 1
                all_infos.append(info)
                generated += duration

        # 可选：对电压幅值进行物理约束剪裁
        final_data[:, :14] = np.clip(final_data[:, :14], 0.85, 1.15)

        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'data': final_data,
                    'labels': final_labels,
                    'attack_infos': all_infos,
                    'normal_data': self.normal_data,
                    'description': '分段混合攻击数据集'
                }, f)

        return final_data, final_labels, all_infos

    def generate_uniform_mixed_dataset(self, total_attack_ratio=0.5,
                                        attack_types=None,
                                        min_duration=80, max_duration=120,
                                        strength_range=(3.0, 8.0),
                                        save_path=None):
            """
            生成均匀混合攻击数据集：整个时间轴上随机生成所有攻击类型的事件，攻击样本比例约为 total_attack_ratio
            """
            if attack_types is None:
                attack_types = ['single_point', 'multi_point', 'slow_drift', 'physical_constrained']

            target_attack_samples = int(self.n_samples * total_attack_ratio)
            final_data = self.normal_data.copy()
            final_labels = np.zeros(self.n_samples, dtype=np.int32)
            all_infos = []

            accumulated = 0
            max_attempts = 2000
            attempts = 0

            while accumulated < target_attack_samples and attempts < max_attempts:
                attempts += 1
                atype = np.random.choice(attack_types)
                start = np.random.randint(0, self.n_samples - max_duration)
                duration = np.random.randint(min_duration, max_duration)
                strength = np.random.uniform(*strength_range)

                if atype == 'single_point':
                    target_feature = np.random.randint(0, self.n_features)
                    attacked, _, info = self.inject_single_point_attack(
                        start, duration, target_feature, strength)
                elif atype == 'multi_point':
                    n_targets = np.random.randint(2, 5)
                    target_features = np.random.choice(self.n_features, n_targets, replace=False)
                    corr = np.random.uniform(0.5, 0.9)
                    attacked, _, info = self.inject_multi_point_attack(
                        start, duration, target_features, corr, strength)
                elif atype == 'slow_drift':
                    target_feature = np.random.randint(0, self.n_features)
                    attacked, _, info = self.inject_slow_drift_attack(
                        start, duration, target_feature, strength)
                else:  # physical_constrained
                    n_buses = np.random.randint(2, 4)
                    target_buses = np.random.choice(range(14), n_buses, replace=False)
                    pattern = np.random.choice(['voltage_drop', 'power_imbalance'])
                    attacked, _, info = self.inject_physical_constrained_attack(
                        start, duration, target_buses, pattern, strength)

                # 可选显著性验证
                # significant, _ = self.validate_attack_significance(attacked, info)
                # if not significant: continue

                # 更新数据
                final_data[start:start+duration] = attacked[start:start+duration]
                final_labels[start:start+duration] = 1
                all_infos.append(info)
                accumulated += duration

            # 可选剪裁
            final_data[:, :14] = np.clip(final_data[:, :14], 0.85, 1.15)

            actual_ratio = np.sum(final_labels) / self.n_samples
            print(f"均匀混合数据集实际攻击比例: {actual_ratio*100:.2f}%")

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    pickle.dump({
                        'data': final_data,
                        'labels': final_labels,
                        'attack_infos': all_infos,
                        'normal_data': self.normal_data,
                        'description': '均匀混合攻击数据集（四种类型随机分布）'
                    }, f)

            return final_data, final_labels, all_infos

    def generate_unsupervised_dataset(self, split_ratio=0.85, total_attack_ratio=0.5,
                                    attack_types=None, min_duration=50, max_duration=200,
                                    strength_range=(3.0, 8.0), save_path=None):
        """
        生成无监督学习所需的数据集：
        - 前 split_ratio 部分完全正常（用于训练+验证）
        - 后 (1-split_ratio) 部分按 total_attack_ratio 注入攻击（用于测试）
        
        参数:
            split_ratio: 训练验证段占总样本的比例
            total_attack_ratio: 测试段中攻击样本的比例（0~1）
            attack_types: 允许的攻击类型列表，若为 None 则使用全部四种
            min_duration, max_duration: 攻击持续时间范围
            strength_range: 攻击强度范围
            save_path: 可选保存路径
        """
        if attack_types is None:
            attack_types = ['single_point', 'multi_point', 'slow_drift', 'physical_constrained']
        
        # 计算分段点
        split_idx = int(self.n_samples * split_ratio)
        train_len = split_idx
        test_len = self.n_samples - split_idx
        
        # 测试段目标攻击样本数
        target_attack = int(test_len * total_attack_ratio)
        
        # 初始化数据
        final_data = self.normal_data.copy()
        final_labels = np.zeros(self.n_samples, dtype=np.int32)
        all_infos = []
        
        accumulated = 0
        max_attempts = 2000
        attempts = 0
    

        while accumulated < target_attack and attempts < max_attempts:
            attempts += 1
            # 1. 先随机选择攻击类型和持续时间
            atype = np.random.choice(attack_types)
            duration = np.random.randint(min_duration, max_duration)
            
            # 2. 检查是否有足够的空间放置该攻击
            max_start = self.n_samples - duration
            if split_idx >= max_start:
                # 没有足够的空间，跳过本次尝试
                continue
            
            # 3. 随机生成起始位置和强度
            start = np.random.randint(split_idx, max_start)
            strength = np.random.uniform(*strength_range)
            
            if atype == 'single_point':
                target_feature = np.random.randint(0, self.n_features)
                attacked, _, info = self.inject_single_point_attack(
                    start, duration, target_feature, strength)
            elif atype == 'multi_point':
                n_targets = np.random.randint(2, 5)
                target_features = np.random.choice(self.n_features, n_targets, replace=False)
                corr = np.random.uniform(0.5, 0.9)
                attacked, _, info = self.inject_multi_point_attack(
                    start, duration, target_features, corr, strength)
            elif atype == 'slow_drift':
                target_feature = np.random.randint(0, self.n_features)
                attacked, _, info = self.inject_slow_drift_attack(
                    start, duration, target_feature, strength)
            else:  # physical_constrained
                n_buses = np.random.randint(2, 4)
                target_buses = np.random.choice(range(14), n_buses, replace=False)
                pattern = np.random.choice(['voltage_drop', 'power_imbalance'])
                attacked, _, info = self.inject_physical_constrained_attack(
                    start, duration, target_buses, pattern, strength)
            
            
            # 更新数据（注意：不要覆盖之前攻击的区域）
            attack_indices = np.arange(start, start+duration)
            # 如果该区间已有攻击（极低概率），跳过
            if np.any(final_labels[attack_indices] == 1):
                continue
            
            final_data[attack_indices] = attacked[attack_indices]
            final_labels[attack_indices] = 1
            all_infos.append(info)
            accumulated += duration
        
        # 可选：电压幅值裁剪
        final_data[:, :14] = np.clip(final_data[:, :14], 0.85, 1.15)
        
        actual_ratio = np.sum(final_labels[split_idx:]) / test_len
        print(f"测试段实际攻击比例: {actual_ratio*100:.2f}% (目标 {total_attack_ratio*100:.2f}%)")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'data': final_data,
                    'labels': final_labels,
                    'attack_infos': all_infos,
                    'normal_data': self.normal_data,
                    'split_ratio': split_ratio,
                    'description': '无监督学习数据集（训练段无攻击，测试段含攻击）'
                }, f)
            print(f"无监督数据集已保存到 {save_path}")
    
        
        return final_data, final_labels, all_infos

def evaluate_state_estimation_impact(net, normal_data, attacked_data, time_step):
    """
    对单个时间步评估攻击对状态估计的影响
    参数:
        net: pandapower网络（已初始化）
        normal_data: 正常数据 (n_timesteps, 56)
        attacked_data: 攻击数据 (n_timesteps, 56)
        time_step: 要评估的时间步索引
    返回:
        dict: 包含估计误差、真实状态等信息
    """
    # 提取该时间步的数据
    normal_meas = normal_data[time_step]
    attacked_meas = attacked_data[time_step]
    
    # 真实状态：正常数据的电压幅值和相角（前28维）
    true_vm = normal_meas[0:14]
    true_va = normal_meas[14:28]
    
    # ---- 对正常数据进行状态估计 ----
    # 清空网络中原有量测（如果有）
    if hasattr(net, 'measurement'):
        net.measurement.drop(net.measurement.index, inplace=True)
    
    # 添加电压幅值量测（所有母线）
    for bus in range(14):
        pp.create_measurement(net, 'v', 'bus', normal_meas[bus], 0.004, bus)  # 标准差设为0.004 p.u.
    # 添加电压相角量测（可选，但需注意参考节点）
    for bus in range(14):
        pp.create_measurement(net, 'va', 'bus', normal_meas[14+bus], 0.1, bus)  # 相角标准差较大
    # 添加注入有功量测（所有母线）
    for bus in range(14):
        pp.create_measurement(net, 'p', 'bus', normal_meas[28+bus], 0.01, bus)
    # 添加注入无功量测（所有母线）
    for bus in range(14):
        pp.create_measurement(net, 'q', 'bus', normal_meas[42+bus], 0.01, bus)
    
    try:
        pp.estimate(net, init='flat')  # 使用平启动
        normal_vm_est = net.res_bus_est.vm_pu.values
        normal_va_est = net.res_bus_est.va_degree.values
        normal_success = True
    except Exception as e:
        print(f"正常数据状态估计失败: {e}")
        normal_success = False
        normal_vm_est = np.nan * np.ones(14)
        normal_va_est = np.nan * np.ones(14)
    
    # ---- 对攻击数据进行状态估计 ----
    net.measurement.drop(net.measurement.index, inplace=True)  # 清空量测
    # 添加攻击后的量测（同样位置）
    for bus in range(14):
        pp.create_measurement(net, 'v', 'bus', attacked_meas[bus], 0.004, bus)
    for bus in range(14):
        pp.create_measurement(net, 'va', 'bus', attacked_meas[14+bus], 0.1, bus)
    for bus in range(14):
        pp.create_measurement(net, 'p', 'bus', attacked_meas[28+bus], 0.01, bus)
    for bus in range(14):
        pp.create_measurement(net, 'q', 'bus', attacked_meas[42+bus], 0.01, bus)
    
    try:
        pp.estimate(net, init='flat')
        attacked_vm_est = net.res_bus_est.vm_pu.values
        attacked_va_est = net.res_bus_est.va_degree.values
        attacked_success = True
    except Exception as e:
        print(f"攻击数据状态估计失败: {e}")
        attacked_success = False
        attacked_vm_est = np.nan * np.ones(14)
        attacked_va_est = np.nan * np.ones(14)
    
    # 计算误差（与真实状态比较）
    if normal_success:
        normal_vm_error = np.abs(normal_vm_est - true_vm)
        normal_va_error = np.abs(normal_va_est - true_va)
    else:
        normal_vm_error = np.nan * np.ones(14)
        normal_va_error = np.nan * np.ones(14)
    
    if attacked_success:
        attacked_vm_error = np.abs(attacked_vm_est - true_vm)
        attacked_va_error = np.abs(attacked_va_est - true_va)
    else:
        attacked_vm_error = np.nan * np.ones(14)
        attacked_va_error = np.nan * np.ones(14)
    
    return {
        'time_step': time_step,
        'true_vm': true_vm,
        'true_va': true_va,
        'normal_vm_est': normal_vm_est,
        'normal_va_est': normal_va_est,
        'attacked_vm_est': attacked_vm_est,
        'attacked_va_est': attacked_va_est,
        'normal_vm_error': normal_vm_error,
        'normal_va_error': normal_va_error,
        'attacked_vm_error': attacked_vm_error,
        'attacked_va_error': attacked_va_error
    }

def load_voltage_dynamics(load_existing=True, data_path="data/normal_data.pkl"):
    """
    验证负荷-电压动态响应，绘制典型母线电压随负荷变化曲线。

    参数:
        load_existing: 若为True且data_path存在，则加载已有数据；否则重新生成数据。
        data_path: 已有正常数据文件路径（pickle格式）。
    """
    generator = PowerSystemDataGenerator(sampling_rate=1/60, total_hours=24)

    print("生成新的正常数据...")
    generator.create_ieee14_network()
    generator.generate_normal_data(save_path=data_path)

    # 绘制负荷-电压动态曲线
    generator.plot_load_voltage_dynamics(save_path='figures/load_voltage_dynamics.png', corr_bus_index=13)


if __name__ == "__main__":

    # 配置参数
    config = {
        'sampling_rate': 1,      # 1Hz采样
        'total_hours': 3,         # 3小时数据
        'window_size': 10,        # 10个时间步的窗口
        'step': 5,               # 滑动步长为5
        'n_attacks': 100,         # 攻击事件数量：生成100个攻击实例
        'min_duration': 90,      # 最小攻击持续时间：每个攻击至少持续80个时间步
        'max_duration': 110,     # 最大攻击持续时间：每个攻击最多持续120个时间步
        'train_ratio': 0.7,      # 70%训练集
        'val_ratio': 0.15        # 15%验证集，15%测试集
    }
    
    # pipeline = FDIA_DataPipeline(config)
    
    # # 运行完整流水线（第一次运行或需要重新生成数据时）single_point', 'multi_point', 'slow_drift','physical_constrained'
    # pipeline.run_full_pipeline(attack_type="multi_point",regenerate=True)

    #  # 获取攻击注入器（pipeline 中已创建 injector 对象）
    # injector = pipeline.injector
    # attacked_data = pipeline.attacked_data
    # attack_infos = pipeline.attack_infos

# ==================== 完整测试：状态估计影响评估 ====================
def state_test():
    import pandapower as pp
    import pandapower.networks as nw
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    # 1. 生成正常数据（使用较小的规模以便快速测试）
    print("="*60)
    print("步骤1: 生成正常数据")
    print("="*60)
    generator = PowerSystemDataGenerator(sampling_rate=1/10, total_hours=1)  # 0.1 Hz, 1小时 → 360个样本
    generator.load_random_std = 0.05   # 适当增大负荷波动
    normal_data, timestamps = generator.generate_normal_data(save_path=None)
    print(f"正常数据形状: {normal_data.shape}")

    # 2. 创建攻击注入器并注入一个单点攻击
    print("\n" + "="*60)
    print("步骤2: 注入单点攻击")
    print("="*60)
    injector = FDIAAttackInjector(normal_data)
    
    # 选择攻击参数：起始时间步200，持续50步，攻击特征0（母线0电压幅值），强度5.0
    start_idx = 200
    duration = 50
    target_feature = 22   # 电压幅值引起状态估计较小
    strength = 5.0

    attacked_data, labels, attack_info = injector.inject_single_point_attack(
        start_idx=start_idx,
        duration=duration,
        target_feature=target_feature,
        strength=strength
    )

    print(f"攻击信息: {attack_info['description']}")
    print(f"攻击区间: [{start_idx}:{start_idx+duration}]")

    # 可选：可视化攻击影响
    # injector.visualize_attack_impact(attacked_data, attack_info, pad_before=100, pad_after=100)

    # 3. 评估攻击对状态估计的影响
    print("\n" + "="*60)
    print("步骤3: 评估状态估计影响")
    print("="*60)

    # 创建一个基准网络（与生成数据时相同）
    base_net = nw.case14()
    from pandapower.estimation import estimate

    # 定义状态估计评估函数（封装在上面的回答中）
    def evaluate_state_estimation_at_timestep(net, normal_meas, attacked_meas, true_vm, true_va):
        """
        对单个时间步进行状态估计并返回误差
        net: 基准网络（会被修改，传入前请复制或重新创建）
        """
        # 清空之前可能存在的量测
        if hasattr(net, 'measurement') and len(net.measurement) > 0:
            net.measurement.drop(net.measurement.index, inplace=True)

        # ---- 正常数据状态估计 ----
        # 添加量测
        for bus in range(14):
            pp.create_measurement(net, 'v', 'bus', normal_meas[bus], 0.004, bus)        # 电压幅值
            pp.create_measurement(net, 'p', 'bus', normal_meas[28+bus], 0.01, bus)      # 有功注入
            pp.create_measurement(net, 'q', 'bus', normal_meas[42+bus], 0.01, bus)      # 无功注入
        # 可选添加相角量测（这里不添加，因为相角量测可能影响收敛，且权重较低）
        try:
            estimate(net, init='flat')
            normal_vm_est = net.res_bus_est.vm_pu.values.copy()
            normal_va_est = net.res_bus_est.va_degree.values.copy()
            normal_success = True
        except Exception as e:
            print(f"正常数据状态估计失败: {e}")
            normal_success = False
            normal_vm_est = np.full(14, np.nan)
            normal_va_est = np.full(14, np.nan)

        # ---- 攻击数据状态估计 ----
        # 重新清空量测
        net.measurement.drop(net.measurement.index, inplace=True)
        for bus in range(14):
            pp.create_measurement(net, 'v', 'bus', attacked_meas[bus], 0.004, bus)
            pp.create_measurement(net, 'p', 'bus', attacked_meas[28+bus], 0.01, bus)
            pp.create_measurement(net, 'q', 'bus', attacked_meas[42+bus], 0.01, bus)
        try:
            estimate(net, init='flat')
            attacked_vm_est = net.res_bus_est.vm_pu.values.copy()
            attacked_va_est = net.res_bus_est.va_degree.values.copy()
            attacked_success = True
        except Exception as e:
            print(f"攻击数据状态估计失败: {e}")
            attacked_success = False
            attacked_vm_est = np.full(14, np.nan)
            attacked_va_est = np.full(14, np.nan)

        # 计算误差
        if normal_success:
            normal_vm_error = np.abs(normal_vm_est - true_vm)
            normal_va_error = np.abs(normal_va_est - true_va)
        else:
            normal_vm_error = np.full(14, np.nan)
            normal_va_error = np.full(14, np.nan)

        if attacked_success:
            attacked_vm_error = np.abs(attacked_vm_est - true_vm)
            attacked_va_error = np.abs(attacked_va_est - true_va)
        else:
            attacked_vm_error = np.full(14, np.nan)
            attacked_va_error = np.full(14, np.nan)

        return {
            'normal_vm_error': normal_vm_error,
            'normal_va_error': normal_va_error,
            'attacked_vm_error': attacked_vm_error,
            'attacked_va_error': attacked_va_error,
            'normal_success': normal_success,
            'attacked_success': attacked_success
        }

    # 获取攻击区间内的真实状态（正常数据的前28维）
    true_vm = normal_data[start_idx:start_idx+duration, 0:14]
    true_va = normal_data[start_idx:start_idx+duration, 14:28]

    # 存储每个时间步的误差
    vm_errors_normal = []
    vm_errors_attacked = []
    va_errors_normal = []
    va_errors_attacked = []
    success_counts = {'normal': 0, 'attacked': 0}

    # 对攻击区间内每个时间步进行评估
    for t in range(start_idx, start_idx+duration):
        # 每个时间步使用一个新的网络副本，避免量测累积
        net_copy = copy.deepcopy(base_net)
        result = evaluate_state_estimation_at_timestep(
            net_copy,
            normal_data[t],
            attacked_data[t],
            true_vm[t-start_idx],
            true_va[t-start_idx]
        )
        vm_errors_normal.append(np.mean(result['normal_vm_error']))
        vm_errors_attacked.append(np.mean(result['attacked_vm_error']))
        va_errors_normal.append(np.mean(result['normal_va_error']))
        va_errors_attacked.append(np.mean(result['attacked_va_error']))
        if result['normal_success']:
            success_counts['normal'] += 1
        if result['attacked_success']:
            success_counts['attacked'] += 1

    # 计算统计指标
    avg_vm_error_normal = np.nanmean(vm_errors_normal)
    avg_vm_error_attacked = np.nanmean(vm_errors_attacked)
    avg_va_error_normal = np.nanmean(va_errors_normal)
    avg_va_error_attacked = np.nanmean(va_errors_attacked)

    print("\n" + "="*60)
    print("状态估计影响统计结果")
    print("="*60)
    print(f"攻击区间长度: {duration} 个时间步")
    print(f"正常数据状态估计成功次数: {success_counts['normal']}/{duration}")
    print(f"攻击数据状态估计成功次数: {success_counts['attacked']}/{duration}")
    print(f"\n电压幅值估计平均误差 (p.u.):")
    print(f"  正常数据: {avg_vm_error_normal:.6f}")
    print(f"  攻击数据: {avg_vm_error_attacked:.6f}")
    print(f"  增长倍数: {avg_vm_error_attacked/avg_vm_error_normal:.2f} 倍")
    print(f"\n电压相角估计平均误差 (度):")
    print(f"  正常数据: {avg_va_error_normal:.6f}")
    print(f"  攻击数据: {avg_va_error_attacked:.6f}")
    print(f"  增长倍数: {avg_va_error_attacked/avg_va_error_normal:.2f} 倍")

    # 可选：绘制误差时间序列对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(vm_errors_normal, label='正常数据', marker='o', linestyle='-')
    plt.plot(vm_errors_attacked, label='攻击数据', marker='s', linestyle='--')
    plt.xlabel('攻击区间内时间步偏移')
    plt.ylabel('电压幅值平均误差 (p.u.)')
    plt.title('电压幅值估计误差对比')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(va_errors_normal, label='正常数据', marker='o', linestyle='-')
    plt.plot(va_errors_attacked, label='攻击数据', marker='s', linestyle='--')
    plt.xlabel('攻击区间内时间步偏移')
    plt.ylabel('电压相角平均误差 (度)')
    plt.title('电压相角估计误差对比')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('state_estimation_impact.png', dpi=150)
    plt.show()

    print("\n测试完成！误差对比图已保存为 'state_estimation_impact.png'")

