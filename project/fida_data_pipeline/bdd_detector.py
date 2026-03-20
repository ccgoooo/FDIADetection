# bdd_detector.py
import numpy as np
import pandapower as pp
from scipy.sparse import csr_matrix, diags
import warnings

class BDDDetector:
    """
    简化的BDD检测器 - 一阶差分异常检测，不是真正的BDD检测
    """
    def __init__(self, network_model, threshold=0.05):
        self.network = network_model
        self.threshold = threshold
        self.baseline = None
        self.baseline_std = None
        
        # 建立基线（使用正常数据）
        self._establish_baseline()
    
    def _establish_baseline(self, n_samples=100):
        """建立正常数据的基线"""
        print("  建立基线...")
        samples = []
        
        for i in range(n_samples):
            try:
                net_copy = self.network.deepcopy()
                pp.runpp(net_copy)
                measurements = self._extract_measurements(net_copy)
                samples.append(measurements)
            except:
                continue
        
        if samples:
            samples = np.array(samples)
            self.baseline = np.mean(samples, axis=0)
            self.baseline_std = np.std(samples, axis=0) + 1e-6  # 避免除零
            print(f"  基线建立完成，特征数: {len(self.baseline)}")
    
    def _extract_measurements(self, net):
        """从网络提取测量值"""
        measurements = []
        
        # 电压幅值
        for bus in net.bus.index:
            measurements.append(net.res_bus.vm_pu[bus])
        
        # 电压相角
        for bus in net.bus.index:
            measurements.append(net.res_bus.va_degree[bus])
        
        # 有功功率
        for bus in net.bus.index:
            measurements.append(net.res_bus.p_mw[bus])
        
        # 无功功率
        for bus in net.bus.index:
            measurements.append(net.res_bus.q_mvar[bus])
        
        return np.array(measurements)
    
    def detect(self, measurements):
        """检测攻击"""
        if self.baseline is None:
            return {'is_attack': False, 'residual_norm': 0}
        
        # 确保维度匹配
        if len(measurements) > len(self.baseline):
            measurements = measurements[:len(self.baseline)]
        elif len(measurements) < len(self.baseline):
            measurements = np.pad(measurements, (0, len(self.baseline) - len(measurements)))
        
        # 计算马氏距离
        diff = measurements - self.baseline
        normalized_diff = diff / self.baseline_std
        residual_norm = np.linalg.norm(normalized_diff) / np.sqrt(len(self.baseline))
        
        is_attack = residual_norm > self.threshold
        
        return {
            'is_attack': is_attack,
            'residual_norm': residual_norm,
            'threshold': self.threshold,
            'suspected_buses': self._find_suspicious(diff)
        }
    
    def _find_suspicious(self, diff):
        """找出可疑的节点"""
        n_buses = len(self.network.bus)
        abs_diff = np.abs(diff)
        top_indices = np.argsort(abs_diff)[-3:]  # 取最大的3个
        
        suspicious = []
        for idx in top_indices:
            if idx < n_buses:
                suspicious.append(f'Bus_{idx+1}_V')
            elif idx < 2*n_buses:
                suspicious.append(f'Bus_{idx - n_buses + 1}_Theta')
            elif idx < 3*n_buses:
                suspicious.append(f'Bus_{idx - 2*n_buses + 1}_P')
            else:
                suspicious.append(f'Bus_{idx - 3*n_buses + 1}_Q')
        
        return suspicious
    """
    基于模型的坏数据检测器 (Bad Data Detection)
    使用加权最小二乘状态估计和残差分析
    """
    
    def __init__(self, network_model, threshold=0.05, method='wls'):
        """
        初始化BDD检测器
        
        参数:
        network_model: pandapower网络模型
        threshold: 检测阈值，残差超过此值判定为攻击
        method: 检测方法 ('wls' - 加权最小二乘, 'ls' - 最小二乘)
        """
        self.network = network_model
        self.threshold = threshold  # 接收threshold参数
        self.method = method
        
        # 构建量测雅可比矩阵
        self.H_matrix = None
        self.R_matrix = None  # 量测协方差矩阵
        self.measurement_indices = {}
        
        # 初始化系统参数
        self._build_system_matrices()
        
        print(f"  BDD检测器初始化完成")
        print(f"    方法: {method}")
        print(f"    阈值: {threshold}")
        print(f"    节点数: {len(network_model.bus)}")
        print(f"    量测量数量: {len(self.measurement_indices)}")
    
    def _build_system_matrices(self):
        """
        构建系统的雅可比矩阵和量测配置
        """
        # 获取节点数量
        n_buses = len(self.network.bus)
        
        # 状态变量: [所有节点的电压幅值, 所有节点的电压相角]
        # 注：平衡节点的相角固定为0，所以实际状态变量数为 2*n_buses - 1
        n_states = 2 * n_buses - 1
        
        # 量测量: 我们使用所有节点的电压幅值、有功注入、无功注入
        measurements = []
        idx = 0
        
        # 电压幅值量测 (每个节点一个)
        for bus in range(n_buses):
            measurements.append(f'V_{bus+1}')
            self.measurement_indices[f'V_{bus+1}'] = idx
            idx += 1
        
        # 有功注入量测 (每个节点一个)
        for bus in range(n_buses):
            measurements.append(f'P_{bus+1}')
            self.measurement_indices[f'P_{bus+1}'] = idx
            idx += 1
        
        # 无功注入量测 (每个节点一个)
        for bus in range(n_buses):
            measurements.append(f'Q_{bus+1}')
            self.measurement_indices[f'Q_{bus+1}'] = idx
            idx += 1
        
        # 初始化雅可比矩阵为稀疏矩阵
        self.H_matrix = csr_matrix((len(measurements), n_states))
        
        # 量测权重矩阵 (假设所有量测精度相同)
        self.R_matrix = diags([1.0] * len(measurements))
        
        print(f"    量测配置: {len(measurements)}个量测量")
        print(f"    状态变量: {n_states}个")
    
    def state_estimation(self, measurements):
        """
        加权最小二乘法状态估计
        
        参数:
        measurements: 量测数据字典或数组
        
        返回:
        x_hat: 估计的状态变量
        """
        # 这里简化实现，实际应该迭代求解
        # 在实际应用中，可以使用pandapower的状态估计功能
        
        # 简单实现：运行潮流计算作为参考
        try:
            # 复制网络
            net_copy = self.network.deepcopy()
            
            # 更新负荷（如果需要）
            if isinstance(measurements, dict):
                # 如果有测量数据，可以用来调整负荷
                pass
            
            # 运行潮流计算
            pp.runpp(net_copy)
            
            # 提取状态估计结果
            n_buses = len(net_copy.bus)
            x_hat = np.zeros(2 * n_buses - 1)
            
            # 电压幅值
            for i in range(n_buses):
                x_hat[i] = net_copy.res_bus.vm_pu.iloc[i]
            
            # 电压相角 (第一个节点作为参考，相角为0)
            for i in range(1, n_buses):
                x_hat[n_buses + i - 1] = net_copy.res_bus.va_degree.iloc[i]
            
            return x_hat
            
        except Exception as e:
            print(f"  状态估计警告: {e}")
            # 返回保守估计
            n_buses = len(self.network.bus)
            x_hat = np.zeros(2 * n_buses - 1)
            x_hat[:n_buses] = 1.0  # 电压设为1.0 pu
            return x_hat
    
    def compute_hx(self, x_hat):
        """
        根据估计状态计算量测估计值 h(x̂)
        
        参数:
        x_hat: 估计的状态变量
        
        返回:
        hx: 量测估计值数组
        """
        n_buses = len(self.network.bus)
        n_measurements = len(self.measurement_indices)
        hx = np.zeros(n_measurements)
        
        # 电压幅值估计
        for i in range(n_buses):
            hx[i] = x_hat[i]
        
        # 有功注入估计 (简化：使用潮流结果)
        try:
            net_copy = self.network.deepcopy()
            # 设置状态变量
            for i in range(n_buses):
                if i < len(net_copy.bus):
                    if i == 0:  # 平衡节点
                        net_copy.bus.at[i, 'vn_kv'] = x_hat[i] * net_copy.bus.at[i, 'vn_kv']
                    else:
                        # 设置PV节点的电压
                        gen_indices = net_copy.gen[net_copy.gen.bus == i].index
                        if len(gen_indices) > 0:
                            net_copy.gen.at[gen_indices[0], 'vm_pu'] = x_hat[i]
            
            pp.runpp(net_copy)
            
            # 有功注入
            for i in range(n_buses):
                hx[n_buses + i] = net_copy.res_bus.p_mw.iloc[i]
            
            # 无功注入
            for i in range(n_buses):
                hx[2*n_buses + i] = net_copy.res_bus.q_mvar.iloc[i]
                
        except:
            # 如果潮流不收敛，使用简单估计
            for i in range(n_buses):
                hx[n_buses + i] = 0
                hx[2*n_buses + i] = 0
        
        return hx
    
    def compute_residual(self, measurements, x_hat=None):
        """
        假设系统缓慢变化——采用上一次的值作为估计
        
        :param self: 说明
        :param measurements: 说明
        :param x_hat: 说明
        """
        # 将输入转换为数组
        if isinstance(measurements, np.ndarray):
            z = measurements[:56]  # 取前56维
        else:
            z = measurements
        
        # 如果没有历史数据，用当前值作为估计
        if not hasattr(self, 'last_z'):
            self.last_z = z
            residual = np.zeros_like(z)
            residual_norm = 0
        else:
            # 用上一次的值作为估计
            residual = z - self.last_z
            residual_norm = np.linalg.norm(residual)
            self.last_z = z
        
        return residual, residual_norm, z
    
    def detect(self, measurements):
        """
        BDD检测主函数
        
        参数:
        measurements: 量测数据
        
        返回:
        检测结果字典
        """
        # 1. 计算残差
        residual, residual_norm, x_hat = self.compute_residual(measurements)
        
        # 2. 阈值判断
        is_attack = residual_norm > self.threshold
        
        # 3. 找出最大残差的位置
        max_residual_idx = np.argmax(np.abs(residual))
        max_residual_value = np.max(np.abs(residual))
        
        # 4. 找出可能的攻击节点
        n_buses = len(self.network.bus)
        suspected_buses = []
        
        if max_residual_idx < n_buses:
            suspected_buses.append(f'Bus_{max_residual_idx+1}_V')
        elif max_residual_idx < 2*n_buses:
            suspected_buses.append(f'Bus_{max_residual_idx - n_buses + 1}_P')
        else:
            suspected_buses.append(f'Bus_{max_residual_idx - 2*n_buses + 1}_Q')
        
        # 5. 返回检测结果
        return {
            'is_attack': is_attack,
            'residual_norm': residual_norm,
            'threshold': self.threshold,
            'max_residual_idx': max_residual_idx,
            'max_residual_value': max_residual_value,
            'suspected_buses': suspected_buses,
            'residual_vector': residual[:10] if len(residual) > 10 else residual,  # 只返回前10个
            'x_hat': x_hat[:10] if len(x_hat) > 10 else x_hat
        }
    
    def set_threshold(self, threshold):
        """设置检测阈值"""
        self.threshold = threshold
        print(f"  检测阈值已更新为: {threshold}")
    
    def get_performance_stats(self, test_data, test_labels):
        """
        评估检测性能
        
        参数:
        test_data: 测试数据
        test_labels: 真实标签
        
        返回:
        性能统计字典
        """
        tp = fp = tn = fn = 0
        
        for i, measurements in enumerate(test_data):
            result = self.detect(measurements)
            pred = 1 if result['is_attack'] else 0
            true = test_labels[i]
            
            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 0:
                tn += 1
            elif pred == 0 and true == 1:
                fn += 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }