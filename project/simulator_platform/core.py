# core/simulator.py
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import Config

class PowerSystemSimulator:
    def __init__(self, config):
        self.config = config
        self.grid = None
        self.current_time = 0
        self.measurements_history = []
        self.attack_labels_history = []
        
    def initialize_grid(self):
        """初始化电网模型"""
        from power_system import GridBuilder
        self.grid = GridBuilder.build_grid(self.config.GRID_MODEL)
        print(f"✅ 已初始化 {self.config.GRID_MODEL} 电网模型")
        
    def simulate_time_step(self, attack_injector=None):
        """
        执行一个时间步的仿真
        """
        # 1. 更新负载（模拟日内变化）
        self._update_load_profile()
        
        # 2. 运行潮流计算
        self._run_power_flow()
        
        # 3. 获取测量数据
        measurements = self._collect_measurements()
        
        # 4. 注入攻击（如果有）
        attack_label = 0
        if attack_injector and np.random.random() < self.config.ATTACK_PROBABILITY:
            attack_type = np.random.choice(self.config.ATTACK_TYPES)
            measurements, attack_info = attack_injector.inject(
                measurements, attack_type, self.current_time
            )
            attack_label = 1
            self._log_attack(attack_info)
        
        # 5. 记录数据
        self.measurements_history.append(measurements)
        self.attack_labels_history.append(attack_label)
        
        # 6. 更新时间
        self.current_time += self.config.TIME_STEP
        
        return measurements, attack_label
    
    def _update_load_profile(self):
        """模拟日内负载变化曲线"""
        hour_of_day = self.current_time % 24
        # 典型日内负载曲线（0-1归一化）
        daily_pattern = 0.6 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # 在原始负载基础上增加随机波动
        for load_idx in range(len(self.grid.load)):
            base_load = self.grid.load.at[load_idx, 'p_mw']
            noise = np.random.uniform(0.95, 1.05)
            self.grid.load.at[load_idx, 'p_mw'] = base_load * daily_pattern * noise
    
    def _run_power_flow(self):
        """运行潮流计算"""
        import pandapower as pp
        try:
            pp.runpp(self.grid)
        except Exception as e:
            print(f"⚠️ 潮流计算失败: {e}")
            # 使用上一次的有效结果
            pass
    
    def _collect_measurements(self):
        """收集测量数据（模拟SCADA/PMU）"""
        measurements = {
            'timestamp': self.current_time,
            'bus_voltages': self.grid.res_bus.vm_pu.values.tolist(),
            'line_loadings': self.grid.res_line.loading_percent.values.tolist(),
            'gen_powers': self.grid.res_gen.p_mw.values.tolist() if len(self.grid.gen) > 0 else [],
        }
        return measurements
    
    def _log_attack(self, attack_info):
        """记录攻击信息"""
        with open(f"{self.config.RESULTS_PATH}attacks_log.csv", 'a') as f:
            f.write(f"{self.current_time},{attack_info['type']},{attack_info['target']}\n")