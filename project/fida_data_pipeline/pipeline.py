import warnings
warnings.filterwarnings('ignore', message='numba cannot be imported')
import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑、黑体
plt.rcParams['axes.unicode_minus'] = False
pd.options.mode.chained_assignment = None  # 关闭链式赋值警告
from data_pipeline import FDIAAttackInjector,PowerSystemDataGenerator,DataNormalizer,SlidingWindowProcessor,FeatureEnhancer84D

class FDIA_DataPipeline:
    """
    FDIA数据处理完整流水线
    集成数据生成、攻击注入、滑动窗口处理
    """
    
    def __init__(self, config=None):
        """
        配置参数：
        config: 配置字典，包含所有参数
        """
        if config is None:
            config = {
                'sampling_rate': 10,  # Hz
                'total_hours': 2,     # 小时
                'window_size': 10,    # 时间步
                'step': 2,           # 滑动步长
                'n_attacks': 10,     # 攻击数量
                'min_duration': 30,  # 最小攻击持续时间
                'max_duration': 100, # 最大攻击持续时间
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'use_feature_enhancement': True,  # 新增：是否使用特征扩展
                'feature_dim': 56                 # 新增：特征维度（扩展后会更新）
            }
        
        self.config = config
        self.generator = None
        self.injector = None
        self.processor = None
        self.normalizer = None
        
        # 数据存储
        self.normal_data = None
        self.attacked_data = None
        self.labels = None
        self.attack_infos = None
        
        # 窗口数据
        self.X_windows = None
        self.y_windows = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    def plot_sample_data(self, data, start_idx=0, duration=200, save_path=None):
        """
        可视化样本数据
        
        参数:
        data: 数据数组
        start_idx: 起始索引
        duration: 持续时间步数
        save_path: 保存路径
        """
        end_idx = min(start_idx + duration, len(data))
        sample_data = data[start_idx:end_idx]
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. 电压幅值 (前14维)
        for i in range(min(5, 14)):
            axes[0].plot(sample_data[:, i], label=f'Bus {i+1}')
        axes[0].set_title('电压幅值 (pu)')
        axes[0].set_ylabel('电压 (pu)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')
        
        # 2. 电压相角 (14-27维)
        for i in range(min(5, 14)):
            axes[1].plot(sample_data[:, 14 + i], label=f'Bus {i+1}')
        axes[1].set_title('电压相角 (度)')
        axes[1].set_ylabel('相角 (度)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        
        # 3. 有功功率 (28-41维)
        for i in range(min(5, 14)):
            axes[2].plot(sample_data[:, 28 + i], label=f'Bus {i+1}')
        axes[2].set_title('有功功率 (MW)')
        axes[2].set_ylabel('功率 (MW)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
        
        # 4. 无功功率 (42-55维)
        for i in range(min(5, 14)):
            axes[3].plot(sample_data[:, 42 + i], label=f'Bus {i+1}')
        axes[3].set_title('无功功率 (MVar)')
        axes[3].set_xlabel('时间步')
        axes[3].set_ylabel('功率 (MVar)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"样本可视化保存到 {save_path}")
        
        plt.show()

    def run_full_pipeline(self, dataset_mode='single', attack_type=None, regenerate=False, output_dir="processed_data"):
        """
        运行完整数据处理流水线
        
        参数：
        regenerate: 是否重新生成数据
        
        dataset_mode: 'single' 生成单一类型攻击（需指定 _test）
                  'mixed'  生成混合攻击数据集
                  'all'    生成所有类型（用于批量生成）
        """
        print("=" * 60)
        print("FDIA数据处理流水线启动")
        print("=" * 60)
        
        # 1. 生成正常数据
        normal_data_path = f"{output_dir}/normal_data.pkl"
        print(f"加载路径: {normal_data_path}")
        
        if regenerate or not os.path.exists(normal_data_path):
            print("\n[阶段1] 生成正常数据...")
            # 使用稳定版本
            self.generator = PowerSystemDataGenerator(
                sampling_rate=self.config['sampling_rate'],
                total_hours=self.config['total_hours']
            )
            self.normal_data, timestamps = self.generator.generate_normal_data(normal_data_path)

            # 可视化样本
            # self.plot_sample_data(
            #     self.normal_data, 
            #     start_idx=0, 
            #     duration=200,
            #     save_path="figures/normal_data_sample.png"
            # )
        else:
            print("\n[阶段1] 加载已有正常数据...")
            normal_data_path = os.path.join(output_dir, "normal_data.pkl")
            print(f"加载路径: {normal_data_path}")
            if os.path.exists(normal_data_path):
                print(f"文件大小: {os.path.getsize(normal_data_path)} 字节")
            else:
                print("文件不存在！")
            with open(normal_data_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.normal_data = data_dict['data']
            # 重新创建 generator 对象，用于获取维度等信息
            self.generator = PowerSystemDataGenerator(
                sampling_rate=self.config['sampling_rate'],
                total_hours=self.config['total_hours']
            )
            print(f"正常数据加载完成: {self.normal_data.shape}")
        
        # 2. 注入攻击
        # 前三种分支是针对监督学习的情况
        if dataset_mode == 'mixed':
            self.generate_mixed_dataset(save_path="data/mixed_dataset.pkl")

        elif dataset_mode == 'uniform_mixed':
                self.generate_uniform_mixed_dataset(total_attack_ratio=0.5, save_path="data/uniform_mixed_dataset.pkl")

        elif dataset_mode == 'single':
        # 原单一类型攻击生成逻辑
            attack_data_path_template = "data/{type}_attack.pkl"
            strength_range = self.config.get('strength_range', (3.0, 8.0))
            self.injector = FDIAAttackInjector(self.normal_data)
            self.attacked_data, self.labels, self.attack_infos = self.injector.generate_specific_attack_dataset(
                attack_type=attack_type,  
                n_attacks=self.config['n_attacks'],
                min_duration=self.config['min_duration'],
                max_duration=self.config['max_duration'],
                strength_range=strength_range,
                save_path_template=attack_data_path_template
            ) 
        # 此分支是针对无监督学习情况
        elif dataset_mode == 'unsupervised':
            # 生成无监督数据集（训练段无攻击，测试段含攻击）
            if attack_type:
                attack_data_path = f"data/unsupervised/{attack_type}_84.pkl"
            else:
                attack_data_path = "data/unsupervised/mixed_84.pkl"

            self.generate_unsupervised_dataset(
                split_ratio=0.85,                     # 前85%为训练+验证
                total_attack_ratio=0.5,                # 测试段攻击比例50%
                attack_type=attack_type,               # 可指定攻击类型，若为None则混合
                save_path=attack_data_path
            )
        print(f"[调试] 攻击后数据形状: {self.attacked_data.shape}")
        
        # 3. 创建滑动窗口
        print("\n[阶段3] 创建滑动窗口...")
        self.processor = SlidingWindowProcessor(
            window_size=self.config['window_size'],
            step=self.config['step'],
            feature_dim=self.generator.get_measurement_dimension()
        )
        
        self.X_windows, self.y_windows = self.processor.create_sliding_windows(
            self.attacked_data, 
            self.labels
        )
        
        print(f"滑动窗口创建完成: {self.X_windows.shape}")
        
        # 分析类别不平衡
        self.processor.analyze_class_imbalance(self.y_windows)
        
        # 可视化窗口样本
        # self.processor.visualize_windows(
        #     self.X_windows, 
        #     self.y_windows,
        #     n_samples=3,
        #     save_path="figures/window_samples.png"
        # )
        
        # 4. 划分数据集
        print("\n[阶段4] 划分训练/验证/测试集...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            self.processor.create_dataset_splits(
                self.X_windows, 
                self.y_windows,
                train_ratio=self.config['train_ratio'],
                val_ratio=self.config['val_ratio']
            )
        
        if self.config.get('use_feature_enhancement', True):
            print("\n[阶段4.5] 特征扩展（56维 → 84维）...")
            self.enhancer = FeatureEnhancer84D()
            
            # 扩展训练集
            print(f"  扩展训练集: {X_train.shape} → ", end="")
            X_train = self.enhancer.enhance_batch(X_train)
            print(f"{X_train.shape}")
            
            # 扩展验证集
            print(f"  扩展验证集: {X_val.shape} → ", end="")
            X_val = self.enhancer.enhance_batch(X_val)
            print(f"{X_val.shape}")
            
            # 扩展测试集
            print(f"  扩展测试集: {X_test.shape} → ", end="")
            X_test = self.enhancer.enhance_batch(X_test)
            print(f"{X_test.shape}")
            
            # 更新特征维度配置
            self.config['feature_dim'] = X_train.shape[2]
            print(f"  特征维度更新: {self.config['feature_dim']}维")

        # 保存扩展后的X_windows（用于后续分析）
        self.X_windows = np.concatenate([X_train, X_val, X_test], axis=0)
        self.y_windows = np.concatenate([y_train, y_val, y_test], axis=0)

        # 5. 数据标准化
        print("\n[阶段5] 数据标准化...")
        self.X_train = X_train  # 更新为扩展后的数据
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        # 5. 数据标准化
        print("\n[阶段5] 数据标准化...")
        self.normalizer = DataNormalizer()
        self.X_train = self.normalizer.fit_transform(self.X_train)
        self.X_val = self.normalizer.transform(self.X_val)
        self.X_test = self.normalizer.transform(self.X_test)
        
        # 保存标准化参数
        self.normalizer.save("models/normalizer.pkl")
        
        # 6. 保存处理后的数据集
        print("\n[阶段6] 保存最终数据集...")
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("数据处理流水线完成!")
        print("=" * 60)
        
        return self
    
    def save_processed_data(self, save_dir="processed_data"):
        """保存处理后的数据集"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为numpy格式
        np.save(f"{save_dir}/X_train.npy", self.X_train)
        np.save(f"{save_dir}/y_train.npy", self.y_train)
        np.save(f"{save_dir}/X_val.npy", self.X_val)
        np.save(f"{save_dir}/y_val.npy", self.y_val)
        np.save(f"{save_dir}/X_test.npy", self.X_test)
        np.save(f"{save_dir}/y_test.npy", self.y_test)
        
        # 保存配置信息
        import json
        with open(f"{save_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"处理后的数据保存到 {save_dir}/")
    
    def load_processed_data(self, load_dir="processed_data"):
        """加载处理后的数据集"""
        self.X_train = np.load(f"{load_dir}/X_train.npy")
        self.y_train = np.load(f"{load_dir}/y_train.npy")
        self.X_val = np.load(f"{load_dir}/X_val.npy")
        self.y_val = np.load(f"{load_dir}/y_val.npy")
        self.X_test = np.load(f"{load_dir}/X_test.npy")
        self.y_test = np.load(f"{load_dir}/y_test.npy")
        
        # 加载配置
        import json
        with open(f"{load_dir}/config.json", 'r') as f:
            self.config = json.load(f)
        
        print(f"处理后的数据从 {load_dir}/ 加载")
        print(f"训练集: {self.X_train.shape}, {self.y_train.shape}")
        print(f"验证集: {self.X_val.shape}, {self.y_val.shape}")
        print(f"测试集: {self.X_test.shape}, {self.y_test.shape}")
        
        return self
    
    def get_data_loaders(self, batch_size=32):
        """获取PyTorch数据加载器"""
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)  # 添加维度
        
        X_val_tensor = torch.FloatTensor(self.X_val)
        y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1)
        
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_test_tensor = torch.FloatTensor(self.y_test).unsqueeze(1)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"数据加载器创建完成:")
        print(f"  训练集: {len(train_loader)}批次, 批次大小={batch_size}")
        print(f"  验证集: {len(val_loader)}批次")
        print(f"  测试集: {len(test_loader)}批次")
        
        return train_loader, val_loader, test_loader

    def run_bdd_detection(self):
        """运行BDD检测作为基准"""
        from bdd_detector import BDDDetector
        
        detector = BDDDetector(self.generator.net)
        
        # 在测试集上评估
        bdd_results = []
        for i in range(len(self.X_test)):
            # 取窗口最后一个时间步作为当前测量值
            measurement = self.X_test[i, -1, :56]  # 原始56维
            result = detector.detect(measurement)
            bdd_results.append(result)
        
        # 与真实标签对比
        self.evaluate_bdd_performance(bdd_results, self.y_test)

    def summarize(self):
        """打印数据流水线摘要"""
        print("\n" + "=" * 60)
        print("FDIA数据流水线摘要")
        print("=" * 60)
        
        print(f"\n1. 原始数据:")
        if self.normal_data is not None:
            print(f"   正常数据形状: {self.normal_data.shape}")
        if self.attacked_data is not None:
            print(f"   攻击数据形状: {self.attacked_data.shape}")
        
        print(f"\n2. 滑动窗口配置:")
        print(f"   窗口大小: {self.config['window_size']}")
        print(f"   滑动步长: {self.config['step']}")
        print(f"   特征维度: {self.config.get('feature_dim', 56)}")
        
        print(f"\n3. 处理后窗口数据:")
        print(f"   总窗口数: {len(self.X_windows) if self.X_windows is not None else 'N/A'}")
        
        print(f"\n4. 数据集划分:")
        if self.X_train is not None:
            print(f"   训练集: {self.X_train.shape}")
            print(f"   验证集: {self.X_val.shape}")
            print(f"   测试集: {self.X_test.shape}")
            
            # 类别分布
            train_attack_ratio = np.sum(self.y_train)/len(self.y_train)*100
            val_attack_ratio = np.sum(self.y_val)/len(self.y_val)*100
            test_attack_ratio = np.sum(self.y_test)/len(self.y_test)*100
            
            print(f"\n5. 类别分布:")
            print(f"   训练集攻击比例: {train_attack_ratio:.2f}%")
            print(f"   验证集攻击比例: {val_attack_ratio:.2f}%")
            print(f"   测试集攻击比例: {test_attack_ratio:.2f}%")
        
        print("\n" + "=" * 60)

    def generate_mixed_dataset(self, split_ratio=0.85,
                            known_types=None,
                            unknown_types=None,
                            total_attack_ratio=0.5,
                            save_path=None):
        """
        生成混合攻击数据集（分段均衡）
        参数含义同之前的 injector.generate_mixed_attack_dataset
        """
        if known_types is None:
            known_types = ['single_point', 'multi_point', 'slow_drift']
        if unknown_types is None:
            unknown_types = ['physical_constrained']

        # 创建攻击注入器
        injector = FDIAAttackInjector(self.normal_data)

        # 调用注入器的方法生成数据
        strength = self.config.get('strength_range', (3.0, 8.0))
        attacked_data, labels, infos = injector.generate_mixed_dataset_with_split(
            split_ratio=split_ratio,
            known_attack_types=known_types,
            unknown_attack_types=unknown_types,
            total_attack_ratio=total_attack_ratio,
            min_duration=self.config['min_duration'],
            max_duration=self.config['max_duration'],
            strength_range=strength,  
            save_path=save_path
        )

        self.attacked_data = attacked_data
        self.labels = labels
        self.attack_infos = infos
        return self
    
    def generate_uniform_mixed_dataset(self, total_attack_ratio=0.5, save_path=None):
        """"
            在所有时间轴上随机生成四类攻击
        """
        injector = FDIAAttackInjector(self.normal_data)
        attacked_data, labels, infos = injector.generate_uniform_mixed_dataset(
            total_attack_ratio=total_attack_ratio,
            min_duration=self.config['min_duration'],
            max_duration=self.config['max_duration'],
            strength_range=(3.0, 8.0),
            save_path=save_path
        )
        self.attacked_data = attacked_data
        self.labels = labels
        self.attack_infos = infos
        return self

    def generate_unsupervised_dataset(self, split_ratio=0.85, total_attack_ratio=0.5,
                                    attack_type=None, save_path=None):
        """
        生成无监督数据集
        """

        if self.normal_data is None:
            raise ValueError("请先加载或生成正常数据")
        
        # 确定攻击类型列表
        if attack_type is None:
            attack_types = ['single_point', 'multi_point', 'slow_drift', 'physical_constrained']
        elif isinstance(attack_type, str):
            attack_types = [attack_type]
        else:
            attack_types = attack_type  # 假设传入的是列表
        
        injector = FDIAAttackInjector(self.normal_data)
        attacked_data, labels, infos = injector.generate_unsupervised_dataset(
            split_ratio=split_ratio,
            total_attack_ratio=total_attack_ratio,
            attack_types=attack_types,
            min_duration=self.config['min_duration'],
            max_duration=self.config['max_duration'],
            strength_range=self.config.get('strength_range', (3.0, 8.0)),
            save_path=save_path
        )
        self.attacked_data = attacked_data
        self.labels = labels
        self.attack_infos = infos


        return self

def supervisedDataset_generation():
    # 公共配置
    base_config = {
        'sampling_rate': 0.15,      # 试验阶段用0.1 Hz
        'total_hours': 24,      # 试验用4
        'window_size': 10,
        'step': 5,
        'n_attacks': 65, 
        'min_duration': 80,
        'max_duration': 120,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'use_feature_enhancement': True,   # 默认开启
    }

    # 生成正常数据（共用）
    output_dir = "processed_data"
    normal_data_path = f"{output_dir}/normal_data.pkl"
    
    print("生成正常数据并保存...")
    generator = PowerSystemDataGenerator(
        sampling_rate=base_config['sampling_rate'],
        total_hours=base_config['total_hours']
    )
    normal_data, timestamps = generator.generate_normal_data(save_path=normal_data_path)

    # 1. 单点攻击数据集（84维）
    pipeline_sp = FDIA_DataPipeline(base_config.copy())
    pipeline_sp.normal_data = normal_data
    pipeline_sp.run_full_pipeline(dataset_mode='single', attack_type='single_point', regenerate=False,output_dir=output_dir)
    pipeline_sp.save_processed_data(save_dir='processed_data/single_point_84')

    # 2. 多点攻击数据集（56维）—— 关闭特征扩展
    config_mp56 = base_config.copy()
    config_mp56['use_feature_enhancement'] = False
    pipeline_mp56 = FDIA_DataPipeline(config_mp56)
    pipeline_mp56.normal_data = normal_data
    pipeline_mp56.run_full_pipeline(dataset_mode='single', attack_type='multi_point', regenerate=False,output_dir=output_dir)
    pipeline_mp56.save_processed_data(save_dir='processed_data/multi_point_56')

    # 3. 多点攻击数据集（84维）—— 开启特征扩展
    pipeline_mp84 = FDIA_DataPipeline(base_config.copy())
    pipeline_mp84.normal_data = normal_data
    pipeline_mp84.run_full_pipeline(dataset_mode='single', attack_type='multi_point', regenerate=False,output_dir=output_dir)
    pipeline_mp84.save_processed_data(save_dir='processed_data/multi_point_84')

    # 4. 漂移攻击数据集（84维）
    pipeline_sd = FDIA_DataPipeline(base_config.copy())
    pipeline_sd.normal_data = normal_data
    pipeline_sd.run_full_pipeline(dataset_mode='single', attack_type='slow_drift', regenerate=False,output_dir=output_dir)
    pipeline_sd.save_processed_data(save_dir='processed_data/slow_drift_84')

    # 5. 物理约束数据集（84维）
    pipeline_pc = FDIA_DataPipeline(base_config.copy())
    pipeline_pc.normal_data = normal_data
    pipeline_pc.run_full_pipeline(dataset_mode='single', attack_type='physical_constrained', regenerate=False,output_dir=output_dir)
    pipeline_pc.save_processed_data(save_dir='processed_data/phys_constrained_84')

    # 6. 混合数据集（84维）
    pipeline_mixed = FDIA_DataPipeline(base_config.copy())
    pipeline_mixed.normal_data = normal_data
    # 需要先确保 FDIAAttackInjector 中有 generate_mixed_attack_dataset 方法
    pipeline_mixed.run_full_pipeline(dataset_mode='mixed', regenerate=False,output_dir=output_dir)
    pipeline_mixed.save_processed_data(save_dir='processed_data/mixed_84')

    # 7. 均匀混合数据集（84维）
    pipeline_uni_mixed = FDIA_DataPipeline(base_config.copy())
    pipeline_uni_mixed.normal_data = normal_data
    pipeline_uni_mixed.run_full_pipeline(dataset_mode='uniform_mixed', regenerate=False,output_dir=output_dir)
    pipeline_uni_mixed.save_processed_data(save_dir='processed_data/uniform_mixed_84')

def unsuperviseDataset_generation():
    base_config = {
        'sampling_rate': 0.2,      
        'total_hours': 72,      
        'window_size': 10,
        'step': 5,
        'n_attacks': 65, 
        'min_duration': 80,
        'max_duration': 120,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'use_feature_enhancement': True,   # 默认开启
    }

        # 生成正常数据（共用）
    output_dir = "processed_data/unsupervised"
    normal_data_path = f"{output_dir}/normal_data.pkl"
    
    print("生成正常数据并保存...")
    generator = PowerSystemDataGenerator(
        sampling_rate=base_config['sampling_rate'],
        total_hours=base_config['total_hours']
    )
    normal_data, timestamps = generator.generate_normal_data(save_path=normal_data_path)
    print("文件大小:", os.path.getsize(normal_data_path))

    # 生成无监督数据集（训练段无攻击，测试段混合攻击）
    pipeline_unsup = FDIA_DataPipeline(base_config)
    pipeline_unsup.normal_data = normal_data

    pipeline_unsup.run_full_pipeline(dataset_mode='unsupervised', attack_type=None, regenerate=False,output_dir=output_dir)
    pipeline_unsup.save_processed_data(save_dir='processed_data/unsupervised/mixed_84')

    # 生成仅包含单点攻击的无监督数据集
    types = ['single_point', 'multi_point', 'slow_drift', 'physical_constrained']
    for atype in types:
        pipeline_unsup_sp = FDIA_DataPipeline(base_config)
        pipeline_unsup_sp.normal_data = normal_data
        pipeline_unsup_sp.run_full_pipeline(dataset_mode='unsupervised', attack_type=atype, regenerate=False,output_dir=output_dir)
        pipeline_unsup_sp.save_processed_data(save_dir=f'processed_data/unsupervised/{atype}_84')

if __name__ == "__main__":
    unsuperviseDataset_generation()