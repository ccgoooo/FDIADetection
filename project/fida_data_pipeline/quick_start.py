"""
快速启动脚本：一键运行数据处理流水线
"""
import sys
import os
from data_pipeline import FDIA_DataPipeline
import torch

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("FDIA数据处理流水线 - 快速启动")
    print("-" * 40)
    
    # 询问用户选择
    print("\n请选择操作:")
    print("1. 生成新数据并运行完整流水线")
    print("2. 加载已有数据")
    print("3. 仅测试数据加载")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    

    
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
    
    else:
        print("无效选择")
        return
    
    # 显示摘要
    pipeline.summarize()
    
    print("\n数据处理完成!")
    print("下一步：使用 train_loader, val_loader, test_loader 进行模型训练")

if __name__ == "__main__":
    main()