# detection_models/integration.py
import numpy as np
import tensorflow as tf
from tensorflow import keras

class DetectionModelInterface:
    """检测模型统一接口"""
    
    def __init__(self, model_type="cnn"):
        self.model_type = model_type
        self.model = self._load_model(model_type)
        
    def _load_model(self, model_type):
        """加载预训练模型"""
        if model_type == "cnn":
            # 加载你的CNN模型
            # model = keras.models.load_model(Config.MODEL_PATHS["cnn"])
            model = self._build_dummy_cnn()  # 临时用虚拟模型
        elif model_type == "dae":
            model = self._build_dummy_dae()
        else:
            model = self._build_dummy_cnn()
        return model
    
    def predict(self, measurements):
        """执行检测"""
        # 将测量数据转换为模型输入格式
        input_data = self._preprocess(measurements)
        
        # 执行预测
        prediction = self.model.predict(input_data, verbose=0)
        
        # 返回检测结果
        return {
            'is_attack': prediction[0][0] > 0.5,
            'confidence': float(prediction[0][0]),
            'timestamp': measurements.get('timestamp', 0)
        }
    
    def _preprocess(self, measurements):
        """数据预处理（根据你的模型需求调整）"""
        # 提取电压数据作为特征向量
        features = np.array(measurements['bus_voltages'])
        
        # 根据模型类型调整维度
        if self.model_type == "cnn":
            # 假设CNN需要二维输入 (1, n_features, 1)
            return features.reshape(1, -1, 1)
        else:
            return features.reshape(1, -1)
    
    def _build_dummy_cnn(self):
        """构建虚拟CNN模型（实际使用时替换为你的模型）"""
        model = keras.Sequential([
            keras.layers.Input(shape=(14, 1)),  # IEEE14有14个节点
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model
    
    def _build_dummy_dae(self):
        """构建虚拟DAE模型"""
        # 类似结构，根据你的DAE模型修改
        return self._build_dummy_cnn()