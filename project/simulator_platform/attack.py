# attack_scenarios/attack_injector.py
import numpy as np

class AttackInjector:
    def __init__(self, grid):
        self.grid = grid
        self.attack_id = 0
        
    def inject(self, measurements, attack_type="random", timestamp=0):
        """
        注入FDIA攻击
        """
        attack_info = {
            'id': self.attack_id,
            'timestamp': timestamp,
            'type': attack_type,
            'target': []
        }
        
        measurements_attacked = measurements.copy()
        
        if attack_type == "random":
            # 随机攻击：随机篡改部分测量值
            num_targets = np.random.randint(1, 4)
            targets = np.random.choice(
                range(len(measurements['bus_voltages'])), 
                num_targets, 
                replace=False
            )
            
            for target in targets:
                measurements_attacked['bus_voltages'][target] *= np.random.uniform(0.8, 1.2)
                attack_info['target'].append(f"BUS_{target+1}")
        
        elif attack_type == "stealthy":
            # 隐蔽攻击：轻微篡改，符合物理规律
            target = np.random.randint(0, len(measurements['bus_voltages']))
            # 缓慢漂移攻击
            drift = 1.0 + 0.05 * np.sin(timestamp / 10.0)  # 最大±5%漂移
            measurements_attacked['bus_voltages'][target] *= drift
            attack_info['target'].append(f"BUS_{target+1}")
        
        elif attack_type == "coordinated":
            # 协同攻击：同时篡改多个相关测量点
            # 选择一条线路的两个端点
            line_idx = np.random.randint(0, len(self.grid.line))
            from_bus = self.grid.line.at[line_idx, 'from_bus']
            to_bus = self.grid.line.at[line_idx, 'to_bus']
            
            # 保持潮流平衡的篡改
            measurements_attacked['bus_voltages'][from_bus] *= 0.9
            measurements_attacked['bus_voltages'][to_bus] *= 1.1
            attack_info['target'].extend([f"BUS_{from_bus+1}", f"BUS_{to_bus+1}"])
        
        self.attack_id += 1
        return measurements_attacked, attack_info
    
    def generate_attack_scenario(self, scenario_name):
        """生成特定攻击场景"""
        scenarios = {
            "single_point": {"type": "random", "duration": 10},
            "slow_drift": {"type": "stealthy", "duration": 60},
            "multi_point": {"type": "coordinated", "duration": 30}
        }
        return scenarios.get(scenario_name, scenarios["single_point"])