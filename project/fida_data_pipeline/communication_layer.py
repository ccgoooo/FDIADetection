# communication_layer.py
import time
import numpy as np
import random
from collections import deque
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SCADACommunicationLayer:
    """
    SCADA通信传输层模拟
    """
    def __init__(self, protocol='iec60870-5-104', latency_ms=100, packet_loss_rate=0.01):
        """
        模拟通信协议：
        - IEC 60870-5-104 (电力系统常用)
        - Modbus
        - DNP3
        """
        self.protocol = protocol
        self.latency = latency_ms / 1000  # 转换为秒
        self.packet_loss_rate = packet_loss_rate
        self.transmission_log = deque(maxlen=10000)  # 使用deque存储日志，限制大小
        self.corruption_rate = 0.005  # 数据包损坏率
        
    def transmit_measurement(self, measurement, timestamp):
        """模拟测量值传输"""
        # 1. 模拟传输延迟
        time.sleep(self.latency + np.random.exponential(0.01))
        
        # 2. 模拟丢包
        if np.random.random() < self.packet_loss_rate:
            self._log_transmission(timestamp, None, status='dropped')  # 记录丢包
            return None  # 丢包
        
        # 3. 模拟数据包损坏
        if np.random.random() < self.corruption_rate:  # 0.5%损坏率
            measurement = self._corrupt_packet(measurement)
            self._log_transmission(timestamp, measurement, status='corrupted')
        else:
            self._log_transmission(timestamp, measurement, status='success')
        
        return measurement
    
    def _corrupt_packet(self, measurement):
        """模拟数据包损坏"""
        if isinstance(measurement, np.ndarray):
            # 随机损坏一个测量值
            corrupted = measurement.copy()
            idx = np.random.randint(len(corrupted))
            # 随机增加或减少50-150%
            corrupted[idx] = corrupted[idx] * (1 + np.random.uniform(-1.5, 1.5))
            return corrupted
        return measurement
    
    def _log_transmission(self, timestamp, data, status='success'):
        """
        记录传输日志
        
        参数:
        timestamp: 时间戳
        data: 传输的数据
        status: 状态 ('success', 'dropped', 'corrupted')
        """
        log_entry = {
            'timestamp': timestamp,
            'status': status,
            'data_size': len(data) if data is not None and hasattr(data, '__len__') else 0,
            'protocol': self.protocol
        }
        
        # 如果是损坏的数据，记录损坏程度
        if status == 'corrupted' and data is not None and hasattr(data, '__len__'):
            log_entry['corruption_level'] = 'partial'
        
        self.transmission_log.append(log_entry)
        
        # 可选：打印日志（调试用）
        # logger.debug(f"传输日志: {log_entry}")
    
    def simulate_network_attack(self, attack_type='man_in_the_middle'):
        """模拟网络层攻击"""
        original_latency = self.latency
        original_loss_rate = self.packet_loss_rate
        original_corruption = self.corruption_rate
        
        if attack_type == 'man_in_the_middle':
            # 中间人攻击：增加延迟和丢包率
            self.latency *= 2
            self.packet_loss_rate *= 3
            self.corruption_rate *= 2
            description = "中间人攻击 - 延迟和丢包率增加"
            
        elif attack_type == 'dos':
            # 拒绝服务攻击：增加大量延迟
            self.latency *= 10
            self.packet_loss_rate = min(self.packet_loss_rate * 5, 0.5)
            description = "DoS攻击 - 高延迟和高丢包"
            
        elif attack_type == 'replay':
            # 重放攻击：这个需要在更高层模拟，这里简单增加延迟
            self.latency *= 1.5
            description = "重放攻击 - 延迟增加"
            
        elif attack_type == 'packet_injection':
            # 数据包注入攻击：增加损坏率
            self.corruption_rate = min(self.corruption_rate * 5, 0.3)
            description = "数据包注入攻击 - 损坏率增加"
            
        else:
            description = f"未知攻击类型: {attack_type}"
        
        return {
            'attack_type': attack_type,
            'description': description,
            'latency_ms': self.latency * 1000,  # 转回ms
            'packet_loss_rate': self.packet_loss_rate * 100,  # 转百分比
            'corruption_rate': self.corruption_rate * 100,  # 转百分比
            'original_latency_ms': original_latency * 1000,
            'original_loss_rate': original_loss_rate * 100,
            'original_corruption': original_corruption * 100
        }
    
    def reset_to_normal(self):
        """重置为正常状态"""
        self.latency = 0.1  # 100ms
        self.packet_loss_rate = 0.01  # 1%
        self.corruption_rate = 0.005  # 0.5%
        logger.info("通信层已重置为正常状态")
    
    def get_statistics(self):
        """获取通信统计信息"""
        total = len(self.transmission_log)
        if total == 0:
            return {
                'total_transmissions': 0,
                'success_rate': 0,
                'dropped_rate': 0,
                'corrupted_rate': 0,
                'avg_latency_ms': self.latency * 1000
            }
        
        successes = sum(1 for log in self.transmission_log if log['status'] == 'success')
        dropped = sum(1 for log in self.transmission_log if log['status'] == 'dropped')
        corrupted = sum(1 for log in self.transmission_log if log['status'] == 'corrupted')
        
        return {
            'total_transmissions': total,
            'success_rate': successes / total * 100,
            'dropped_rate': dropped / total * 100,
            'corrupted_rate': corrupted / total * 100,
            'avg_latency_ms': self.latency * 1000,
            'packet_loss_rate': self.packet_loss_rate * 100,
            'corruption_rate': self.corruption_rate * 100
        }
    
    def print_log_summary(self):
        """打印日志摘要"""
        stats = self.get_statistics()
        print("\n=== 通信层统计 ===")
        print(f"总传输次数: {stats['total_transmissions']}")
        print(f"成功率: {stats['success_rate']:.2f}%")
        print(f"丢包率: {stats['dropped_rate']:.2f}%")
        print(f"损坏率: {stats['corrupted_rate']:.2f}%")
        print(f"平均延迟: {stats['avg_latency_ms']:.2f}ms")
        print("=" * 20)