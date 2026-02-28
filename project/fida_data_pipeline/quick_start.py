"""
快速启动脚本：一键运行数据处理流水线
"""
import sys
import os
import threading  
import time
from datetime import datetime
import torch
from data_pipeline import FDIA_DataPipeline
import warnings
import numpy as np

warnings.filterwarnings('ignore', message='numba cannot be imported')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_complete_pipeline():
    """运行完整的基础仿真链路"""
    print("\n" + "=" * 60)
    print("启动完整实时检测链路")
    print("=" * 60)
    
    try:
        # 1. 创建仪表板
        print("\n[1/4] 初始化仪表板...")
        from dashboard import RealtimeDashboard
        dashboard = RealtimeDashboard()
        
        # 2. 启动仪表板线程
        print("  启动仪表板线程...")
        dashboard_thread = threading.Thread(
            target=dashboard.run, 
            kwargs={'debug': False, 'port': 8050},
            daemon=True
        )
        dashboard_thread.start()
        print(f"  仪表板已启动，请访问 http://localhost:8050")
        
        # 3. 初始化其他组件
        print("\n[2/4] 初始化数据生成器...")
        from data_pipeline import PowerSystemDataGenerator
        generator = PowerSystemDataGenerator(sampling_rate=10, total_hours=24)
        generator.create_ieee14_network()
        
        print("\n[3/4] 初始化通信层...")
        from communication_layer import SCADACommunicationLayer
        comm_layer = SCADACommunicationLayer(
            protocol='iec60870-5-104',
            latency_ms=100,
            packet_loss_rate=0.01
        )
        
        print("\n[4/4] 初始化BDD检测器...")
        from bdd_detector import BDDDetector
        bdd_detector = BDDDetector(
            network_model=generator.net,
            threshold=100.0
        )
        
        print("\n" + "=" * 60)
        print("系统初始化完成，开始实时检测...")
        print("按 Ctrl+C 停止")
        print("=" * 60)
        
        # 4. 模拟实时数据流
        start_time = datetime.now()
        sample_count = 0
        attack_count = 0
        
        while True:
            # 生成数据
            current_time = datetime.now()
            measurement = generator.run_power_flow(current_time)

            # 通信传输
            received_data = comm_layer.transmit_measurement(measurement, current_time)
            
            if received_data is not None:
                # BDD检测
                bdd_result = bdd_detector.detect(received_data)
                
                # 更新仪表板
                dashboard.update_data({
                    'timestamp': current_time,
                    'measurement': received_data,
                    'bdd_result': bdd_result
                })
                
                # 统计
                sample_count += 1
                if bdd_result['is_attack']:
                    attack_count += 1
                
                # 每10个样本打印一次状态
                if sample_count % 10 == 0:
                    print(f"  样本数: {sample_count}, 攻击: {attack_count}, "
                          f"残差: {bdd_result['residual_norm']:.4f}")
            
            # 控制采样率 (10Hz)
            time.sleep(0.1)
            
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("请确保所有必要的文件都存在:")
        print("  - dashboard.py")
        print("  - communication_layer.py")
        print("  - bdd_detector.py")
    except KeyboardInterrupt:
        print("\n\n✅ 实时检测已停止")
        print(f"总计处理样本数: {sample_count}")
        print(f"检测到攻击次数: {attack_count}")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("FDIA数据处理流水线 - 快速启动")
    print("-" * 40)
    
    # 询问用户选择
    print("\n请选择操作:")
    print("1. 生成新数据并运行完整流水线")
    print("2. 加载已有数据")
    print("3. 仅测试数据加载")
    print("4. 运行完整实时检测链路")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    # 默认配置
    config = {
        'sampling_rate': 10,
        'total_hours': 1,  # 1小时数据用于快速测试
        'window_size': 10,
        'step': 2,
        'n_attacks': 5,
        'min_duration': 30,
        'max_duration': 100,
        'train_ratio': 0.7,
        'val_ratio': 0.15
    }
    
    pipeline = FDIA_DataPipeline(config)
    
    if choice == '1':
        print("\n开始生成新数据...")
        pipeline.run_full_pipeline(regenerate=True)
        
    elif choice == '2':
        print("\n加载已有数据...")
        try:
            pipeline.load_processed_data("processed_data")
            print("数据加载成功!")
        except Exception as e:
            print(f"加载失败: {e}")
            print("请先运行选项1生成数据")
            return
    
    elif choice == '3':
        print("\n测试数据加载...")
        try:
            pipeline.load_processed_data("processed_data")
            train_loader, val_loader, test_loader = pipeline.get_data_loaders(batch_size=16)
            
            # 测试一个批次
            for batch_X, batch_y in train_loader:
                print(f"\n测试批次:")
                print(f"  输入形状: {batch_X.shape}")
                print(f"  标签形状: {batch_y.shape}")
                print(f"  攻击样本数: {torch.sum(batch_y==1).item()}/{batch_y.shape[0]}")
                break
                
        except Exception as e:
            print(f"测试失败: {e}")
            return
    
    elif choice == '4':
        run_complete_pipeline()
        return
    
    else:
        print("无效选择")
        return
    
    # 显示摘要
    pipeline.summarize()
    
    print("\n数据处理完成!")
    print("下一步：使用 train_loader, val_loader, test_loader 进行模型训练")

if __name__ == "__main__":
    main()