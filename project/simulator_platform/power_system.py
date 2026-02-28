# power_system/grid_builder.py
import pandapower as pp
import pandapower.networks as nw
import numpy as np

class GridBuilder:
    @staticmethod
    def build_grid(model_name="ieee14"):
        """构建标准测试电网"""
        if model_name == "ieee9":
            net = nw.case9()
        elif model_name == "ieee14":
            net = nw.case14()
        elif model_name == "ieee30":
            net = nw.case30()
        else:
            raise ValueError(f"未知的电网模型: {model_name}")
        
        # 添加测量点标记（便于后续定位）
        for i, bus in enumerate(net.bus):
            net.bus.at[i, 'measurement_id'] = f"BUS_{i+1}"
        
        return net
    
    @staticmethod
    def add_distributed_generation(net, penetration=0.2):
        """添加分布式电源（光伏/风电）"""
        # 随机选择部分负载节点添加DG
        dg_buses = np.random.choice(
            net.load.bus.values,
            size=int(len(net.load) * penetration),
            replace=False
        )
        
        for bus in dg_buses:
            pp.create_sgen(net, bus, p_mw=net.load.loc[net.load.bus==bus, 'p_mw'].values[0]*0.3)
        
        return net