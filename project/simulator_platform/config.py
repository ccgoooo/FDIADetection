# config/settings.py
import os

class Config:
    # 仿真参数
    SIMULATION_TIME = 24  # 仿真时长（小时）
    TIME_STEP = 1.0       # 时间步长（秒）
    GRID_MODEL = "ieee14" # 电网模型：ieee9, ieee14, ieee30
    
    # 攻击参数
    ATTACK_PROBABILITY = 0.3  # 攻击注入概率
    ATTACK_TYPES = ["random", "stealthy", "coordinated"]
    
    # 数据存储
    DATA_PATH = "./data/"
    RESULTS_PATH = "./results/"
    
    # 模型路径
    MODEL_PATHS = {
        "cnn": "./models/cnn_model.h5",
        "dae": "./models/dae_model.h5",
        "ae_lstm": "./models/ae_lstm_model.h5"
    }

    # 数据生成参数
    GENERATE_DATA_MODE = True  # 启用数据生成模式
    DATA_SAMPLES = 10000       # 生成数据样本数
    NORMAL_RATIO = 0.7         # 正常数据比例
    ATTACK_RATIO = 0.0         # 攻击数据比例
    
    # 攻击类型分布
    ATTACK_DISTRIBUTION = {
        "random": 0.4,     # 40%随机攻击
        "stealthy": 0.3,   # 30%隐蔽攻击  
        "coordinated": 0.3 # 30%协同攻击
    }