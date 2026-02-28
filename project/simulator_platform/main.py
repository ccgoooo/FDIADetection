# main.py
import os
import time
import json
from config import Config
from core import PowerSystemSimulator
from power_system import GridBuilder
from attack import AttackInjector
from detect import DetectionModelInterface
from evaluation import Evaluator
from visualization import ResultVisualizer

def main():
    # 0. 初始化配置和目录
    config = Config()
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    print("=" * 60)
    print("FDIA检测仿真平台启动")
    print(f"电网模型: {config.GRID_MODEL}")
    print(f"仿真时长: {config.SIMULATION_TIME} 小时")
    print("=" * 60)
    
    # 1. 初始化仿真器
    simulator = PowerSystemSimulator(config)
    simulator.initialize_grid()
    
    # 2. 初始化攻击注入器
    attack_injector = AttackInjector(simulator.grid)
    
    # 3. 初始化检测模型（选择你的模型）
    detector = DetectionModelInterface(model_type="cnn")
    
    # 4. 初始化评估器
    evaluator = Evaluator()
    
    # 5. 开始仿真循环
    start_time = time.time()
    
    print("\n🚀 开始仿真...")
    while simulator.current_time < config.SIMULATION_TIME * 3600:  # 转换为秒
        # 执行一个时间步
        measurements, true_label = simulator.simulate_time_step(attack_injector)
        
        # 运行检测模型
        detection_result = detector.predict(measurements)
        pred_label = 1 if detection_result['is_attack'] else 0
        
        # 记录结果
        evaluator.add_result(
            true_label=true_label,
            pred_label=pred_label,
            detection_time=simulator.current_time
        )
        
        # 实时输出（每100步输出一次）
        if simulator.current_time % 100 == 0:
            print(f"时间: {simulator.current_time/3600:.1f}h | "
                  f"真实: {'攻击' if true_label else '正常'} | "
                  f"检测: {'攻击' if pred_label else '正常'} | "
                  f"置信度: {detection_result['confidence']:.3f}")
        
        # 控制仿真速度（可调整）
        # time.sleep(0.01)  # 实时仿真时使用
    
    # 6. 仿真结束，进行评估
    print("\n✅ 仿真完成!")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    
    # 生成评估报告
    metrics = evaluator.generate_report()
    
    # 保存数据
    print("\n💾 保存数据中...")
    with open(f"{config.RESULTS_PATH}measurements.json", 'w') as f:
        json.dump(simulator.measurements_history, f)
    
    with open(f"{config.RESULTS_PATH}labels.json", 'w') as f:
        json.dump({
            'true': simulator.attack_labels_history,
            'pred': [1 if evaluator.pred_labels[i] else 0 for i in range(len(evaluator.pred_labels))]
        }, f)
    
    # 7. 可视化结果
    print("\n📊 生成可视化图表...")
    ResultVisualizer.plot_real_time_dashboard(
        simulator.measurements_history[:1000],  # 只显示前1000个点
        simulator.attack_labels_history[:1000],
        evaluator.pred_labels[:1000]
    )
    
    print(f"\n🎉 所有结果已保存到 {config.RESULTS_PATH} 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()