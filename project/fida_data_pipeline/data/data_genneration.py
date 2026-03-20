import numpy as np
import pandapower as pp
import pandapower.networks as nw
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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
        self.base_loads = {}            # 初始化base_loads
        self.load_history = []          # 记录每个时间步的负荷值 (n_samples, n_loads)
        self.convergence_history = []   # 记录收敛标志 (bool)
        
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
                random_factor = 1 + np.random.normal(0, 0.05)  # ±2%波动
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


class DataEvaluator:
    def __init__(self, generator, normal_data, timestamps, 
                 attacked_data=None, labels=None, attack_infos=None):
        self.generator = generator
        self.normal = normal_data
        self.timestamps = timestamps
        self.attacked = attacked_data
        self.labels = labels
        self.attack_infos = attack_infos
        # 从生成器获取基准测量值
        self.base_measurements = generator.get_base_measurements()

    def plot_normal_stats(self, save_path=None):
        """绘制正常数据与基准值的统计对比"""
        stats = {
            'mean': np.mean(self.normal, axis=0),
            'std': np.std(self.normal, axis=0),
            'min': np.min(self.normal, axis=0),
            'max': np.max(self.normal, axis=0),
            'base': self.base_measurements
        }
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        feature_names = ['电压幅值 (pu)', '电压相角 (°)', '有功功率 (MW)', '无功功率 (Mvar)']
        for i in range(4):
            ax = axes[i]
            idx = range(i*14, (i+1)*14)
            x = np.arange(14)
            ax.plot(x, stats['base'][idx], 'o-', label='基准值', linewidth=2)
            ax.plot(x, stats['mean'][idx], 's-', label='生成数据均值')
            ax.fill_between(x, stats['min'][idx], stats['max'][idx], alpha=0.3, label='生成数据范围')
            ax.set_title(feature_names[i])
            ax.set_xlabel('母线编号')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return stats

    def plot_load_profiles(self):
        """绘制各负荷节点的有功功率曲线（需 generator 有 load_history）"""
        if not hasattr(self.generator, 'load_history') or not self.generator.load_history:
            print("无负荷历史记录，无法绘制")
            return
        load_history = np.array(self.generator.load_history)  # (n_samples, n_loads)
        hours = np.array([t.hour + t.minute/60 for t in self.timestamps])
        plt.figure(figsize=(14, 6))
        for load_idx in range(load_history.shape[1]):
            plt.plot(hours, load_history[:, load_idx], alpha=0.7, label=f'负荷 {load_idx}')
        plt.xlabel('小时')
        plt.ylabel('有功功率 (MW)')
        plt.title('各负荷节点日负载曲线')
        plt.grid(True)
        plt.legend(loc='upper right', ncol=4, fontsize=8)
        plt.tight_layout()
        plt.show()

    def convergence_statistics(self):
        """统计潮流收敛率"""
        if not hasattr(self.generator, 'convergence_history'):
            print("无收敛历史记录")
            return
        conv = np.array(self.generator.convergence_history)
        total = len(conv)
        converged = np.sum(conv)
        print(f"潮流计算总次数: {total}")
        print(f"收敛次数: {converged} ({converged/total*100:.2f}%)")
        # 可绘制收敛标志的分布
        plt.figure(figsize=(10,2))
        plt.plot(conv.astype(int), drawstyle='steps-post')
        plt.yticks([0,1], ['失败','成功'])
        plt.xlabel('时间步')
        plt.title('潮流收敛状态')
        plt.grid(True)
        plt.show()

    def compute_anomaly_scores(self, method='zscore'):
        """计算每个时间点的异常分数（仅当提供了 attacked_data 时可用）"""
        if self.attacked is None:
            print("未提供攻击数据")
            return None
        if method == 'zscore':
            mean = np.mean(self.normal, axis=0)
            std = np.std(self.normal, axis=0)
            z = np.abs((self.attacked - mean) / (std + 1e-8))
            score = np.mean(z, axis=1)   # 或 np.max(z, axis=1)
        elif method == 'mahalanobis':
            # 计算协方差和逆矩阵
            cov = np.cov(self.normal.T)
            inv_cov = np.linalg.pinv(cov)
            mean = np.mean(self.normal, axis=0)
            delta = self.attacked - mean
            score = np.array([np.sqrt(d @ inv_cov @ d) for d in delta])
        else:
            raise ValueError("method 必须为 'zscore' 或 'mahalanobis'")

        # 绘制分数及攻击标签
        plt.figure(figsize=(12,4))
        plt.plot(score, label='异常分数')
        if self.labels is not None:
            plt.fill_between(range(len(score)), 0, score*self.labels, alpha=0.3, color='red', label='攻击时段')
        plt.axhline(y=np.percentile(score[self.labels==0], 95) if self.labels is not None else np.percentile(score, 95),
                    color='k', linestyle='--', label='95% 分位数')
        plt.xlabel('时间步')
        plt.ylabel('异常分数')
        plt.legend()
        plt.title(f'异常分数 ({method})')
        plt.show()
        return score
    
    def check_voltage_limits(self, data=None, label=''):
        """检查电压幅值是否越限（默认0.95~1.05 pu）"""
        if data is None:
            data = self.normal
        vm = data[:, :14]   # 前14个特征为电压幅值
        low_violations = np.sum(vm < 0.95, axis=1)
        high_violations = np.sum(vm > 1.05, axis=1)
        total_violations = low_violations + high_violations

        plt.figure(figsize=(12,4))
        plt.plot(total_violations, label='越限母线数')
        plt.xlabel('时间步')
        plt.ylabel('越限母线数量')
        plt.title(f'电压越限情况 {label}')
        plt.grid(True)
        plt.legend()
        plt.show()
        print(f"平均越限母线数: {np.mean(total_violations):.2f}")
        print(f"最大越限母线数: {np.max(total_violations)}")

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
    
    def inject_single_point_attack(self, start_idx, duration, target_feature,strength=3.0):
        """
        注入单点突增攻击
        
        参数：
        start_idx: 攻击开始位置
        duration: 攻击持续时间（采样点数）
        target_feature: 目标特征索引（0-55）
        magnitude: 攻击强度（相对于正常值的比例）
        """
        attacked_data = self.normal_data.copy()
        end_idx = min(start_idx + duration, self.n_samples)
        feat_std = np.std(self.normal_data[:, target_feature])
        # 使用加法形式，偏差 = strength * feat_std * 随机符号（或固定正）
        attack_bias = np.random.choice([-1, 1]) * strength * feat_std
        attacked_data[start_idx:end_idx, target_feature] += attack_bias

        attack_labels = np.zeros(self.n_samples, dtype=np.int32)
        attack_labels[start_idx:end_idx] = 1
        
        # 攻击详情
        attack_info = {
            'type': 'single_point',
            'start_idx': start_idx,
            'duration': duration,
            'target_feature': target_feature,
            'strength': strength,  # 记录强度倍数
            'affected_features': [target_feature],
            'description': f'单点攻击：特征{target_feature}在[{start_idx}:{end_idx}]，强度{strength}σ'
        }
        print(f"[DEBUG] single_point: strength={strength}, feat_std={feat_std:.6f}, bias={attack_bias:.6f}")
        print(f"[DEBUG] 修改的特征 {target_feature} 在区间 [{start_idx}:{end_idx}] 的均值变化: {np.mean(attacked_data[start_idx:end_idx, target_feature] - self.normal_data[start_idx:end_idx, target_feature]):.6f}")
        
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

        print(f"[DEBUG] single_point: strength={strength}, feat_std={feat_std:.6f}, bias={attack_bias:.6f}")
        print(f"[DEBUG] 修改的特征 {target_features} 在区间 [{start_idx}:{end_idx}] 的均值变化: {np.mean(attacked_data[start_idx:end_idx, target_features] - self.normal_data[start_idx:end_idx, target_features]):.6f}")

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

        print(f"[DEBUG] physical_constrained: affected_features = {target_feature}")
        return attacked_data, attack_labels, attack_info
    
    def inject_physical_constrained_attack(self, start_idx, duration, target_buses, attack_pattern='voltage_drop', strength=3.0):
        attacked_data = self.normal_data.copy()
        end_idx = min(start_idx + duration, self.n_samples)
        affected = []  # 初始化受影响特征列表

        if attack_pattern == 'voltage_drop':
            for bus_idx in target_buses:
                voltage_feature = bus_idx
                feat_std = np.std(self.normal_data[:, voltage_feature])
                # 生成平滑下降曲线，最终下降 strength * feat_std
                drop_curve = np.linspace(0, strength * feat_std, end_idx - start_idx)
                attacked_data[start_idx:end_idx, voltage_feature] -= drop_curve
                affected.append(voltage_feature)  # 记录受影响特征

                # 相邻总线也受影响，幅度减半
                if bus_idx < 13:
                    neighbor_feature = bus_idx + 1
                    neighbor_std = np.std(self.normal_data[:, neighbor_feature])
                    neighbor_drop = np.linspace(0, strength * 0.5 * neighbor_std, end_idx - start_idx)
                    attacked_data[start_idx:end_idx, neighbor_feature] -= neighbor_drop
                    affected.append(neighbor_feature)

        elif attack_pattern == 'power_imbalance':
            if len(target_buses) >= 2:
                gen_bus = target_buses[0]
                load_bus = target_buses[1]

                # 发电总线：增加有功功率
                gen_feature = 28 + gen_bus
                gen_std = np.std(self.normal_data[:, gen_feature])
                gen_increase = np.linspace(0, strength * gen_std, end_idx - start_idx)
                attacked_data[start_idx:end_idx, gen_feature] += gen_increase
                affected.append(gen_feature)

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
        
        print(f"[DEBUG] single_point: strength={strength}, feat_std={feat_std:.6f}, load_decrease={load_decrease:.6f}")
        print(f"[DEBUG] 修改的特征 {target_buses} 在区间 [{start_idx}:{end_idx}] 的均值变化: {np.mean(attacked_data[start_idx:end_idx, target_buses] - self.normal_data[start_idx:end_idx, target_buses]):.6f}")



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
                                   max_duration=200, strength_range=(3.0, 6.0), save_path_template="data/attack_{type}.pkl"):
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
            print(f"[DEBUG] 特征 {feat}: ratio={ratio:.2f}, feat_std={feat_std:.6f}, use_max={use_max}")
        
        return overall_significant, details

    def visualize_attack_impact(self, attacked_data, attack_info, save_path=None):
        """
        可视化单个攻击的影响
        """
        start = attack_info['start_idx']
        end = start + attack_info['duration']
        
        if 'target_feature' in attack_info:
            features = [attack_info['target_feature']]
        elif 'target_features' in attack_info:
            features = attack_info['target_features']
        else:
            features = list(range(min(5, self.n_features)))  # 默认前5个
        
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 2, figsize=(12, 3*n_features))
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        for i, feat in enumerate(features):
            normal_seg = self.normal_data[start:end, feat]
            attack_seg = attacked_data[start:end, feat]
            time = np.arange(start, end)
            
            # 左图：时间序列对比
            axes[i, 0].plot(time, normal_seg, label='正常', alpha=0.7)
            axes[i, 0].plot(time, attack_seg, label='攻击', alpha=0.7)
            axes[i, 0].axvspan(start, end, alpha=0.2, color='red')
            axes[i, 0].set_title(f'特征 {feat} 时间序列')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # 右图：分布对比（箱线图或直方图）
            axes[i, 1].boxplot([normal_seg, attack_seg], labels=['正常', '攻击'])
            axes[i, 1].set_title(f'特征 {feat} 分布对比')
            axes[i, 1].grid(True, alpha=0.3)
            
            # 标注显著性
            if 'significance_details' in attack_info and feat in attack_info['significance_details']:
                ratio = attack_info['significance_details'][feat]['ratio']
                axes[i, 0].text(0.02, 0.95, f'ratio={ratio:.2f}σ', transform=axes[i, 0].transAxes,
                            bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

if __name__ == "__main__":
    # 示例：生成1小时正常数据，采样率5Hz
    generator = PowerSystemDataGenerator(sampling_rate=5, total_hours=1.5)
    normal_data, timestamps = generator.generate_normal_data(save_path="data/normal_data.pkl")

    print("负荷历史前10个样本:")
    print(np.array(generator.load_history)[:10])
    
    # # 创建评估器并评估正常数据
    # evaluator = DataEvaluator(generator, normal_data, timestamps)
    # evaluator.plot_normal_stats(save_path="data/normal_stats.png")
    # evaluator.plot_load_profiles()
    # evaluator.convergence_statistics()
    # evaluator.check_voltage_limits(data=normal_data, label='正常数据')
    

    # voltage_features = normal_data[:, :14]  # 前14维是电压幅值
    # std_per_bus = np.std(voltage_features, axis=0)
    # np.set_printoptions(precision=6, suppress=True)
    # print("各节点电压标准差(p.u.):", std_per_bus)
    # print("平均标准差:", np.mean(std_per_bus))
    

    # # 生成攻击数据
    injector = FDIAAttackInjector(normal_data)
    attacked_data, labels, infos = injector.generate_attack_dataset(
        n_attacks=10, min_duration=50, max_duration=100,
        strength_range=(5.0, 10.0), save_path="data/attack_data.pkl"
    )

    injector = FDIAAttackInjector(normal_data)
    test_start = 1000
    test_duration = 50
    test_strength = 8.0
    test_feature = 0
    attacked, labels, info = injector.inject_single_point_attack(test_start, test_duration, test_feature, test_strength)
