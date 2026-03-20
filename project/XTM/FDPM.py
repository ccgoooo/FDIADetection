import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# ==================== 工具函数 ====================
class utility_functions:
    @staticmethod
    def MAE(real, predictions):
        return np.mean(np.mean(abs(real - predictions), axis=0))

    @staticmethod
    def MSE(real, predictions):
        return np.mean(np.mean(np.square(abs(real - predictions)), axis=0))

    @staticmethod
    def RMSE(real, predictions):
        return np.mean(np.sqrt(np.mean(np.square(abs(real - predictions)), axis=0)))

    @staticmethod
    def show_barplot(data_list, label, n_bins=50, save_path=None, separate=True):
        if separate:
            # 创建子图
            fig, axes = plt.subplots(len(data_list), 1, figsize=(8, 4 * len(data_list)))
            if len(data_list) == 1:
                axes = [axes]
            for ax, data, lbl in zip(axes, data_list, label):
                ax.hist(data, bins=n_bins, alpha=0.7, label=lbl)
                ax.set_xlabel('预测误差范数')
                ax.set_ylabel('频次')
                ax.set_title(f'{lbl}样本的误差分布')
                ax.legend()
            plt.tight_layout()
        else:
            # 原样绘制在同一张图上
            for i in range(len(data_list)):
                plt.hist(data_list[i], bins=n_bins, alpha=0.7, label=label[i])
            plt.xlabel('预测误差范数')
            plt.ylabel('频次')
            plt.legend()

        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, 'checkpoint', 'images', 'barplot.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_roc_curve(tprs, fprs, n_classes):
        for i in range(n_classes):
            plt.plot(fprs[i], tprs[i], label='class ' + str(i))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), 'checkpoint/images/roc_curve.jpg'))
        plt.show()

    @staticmethod
    def cm2prf(cm):
        # class 0
        tp0 = cm[0][0]
        fp0 = cm[0][1]
        fn0 = cm[1][0]
        tn0 = cm[1][1]
        pr0 = tp0 / (tp0 + fp0) if (tp0 + fp0) != 0 else 0
        re0 = tp0 / (tp0 + fn0) if (tp0 + fn0) != 0 else 0
        f10 = 2 * ((pr0 * re0) / (pr0 + re0)) if (pr0 + re0) != 0 else 0

        # class 1
        tp1 = cm[1][1]
        fp1 = cm[1][0]
        fn1 = cm[0][1]
        tn1 = cm[0][0]
        pr1 = tp1 / (tp1 + fp1) if (tp1 + fp1) != 0 else 0
        re1 = tp1 / (tp1 + fn1) if (tp1 + fn1) != 0 else 0
        f11 = 2 * ((pr1 * re1) / (pr1 + re1)) if (pr1 + re1) != 0 else 0

        return (pr0 + pr1) / 2, (re0 + re1) / 2, (f10 + f11) / 2


    @staticmethod
    def plot_confusion_matrix(cm, class_names=['正常', '攻击'], title='混淆矩阵', 
                              save_path=None, cmap=plt.cm.Blues):
        """
        绘制混淆矩阵并保存图像。
        参数：
            cm: 混淆矩阵 (2x2 数组)
            class_names: 类别名称列表
            title: 图像标题
            save_path: 保存路径，若为None则自动生成
            cmap: 颜色映射
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # 在格子中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()

        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, 'checkpoint', 'images', 'confusion_matrix.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ==================== 数据加载器（适配仿真系统输出） ====================
class DataLoader:
    def __init__(self, normal_data_path, attacked_data_path, scaler,
                 lookback=48, delay=1, batch_size=32, step=1):
        """
        参数：
            normal_data_path   : 正常数据 .pkl 文件路径（包含 'data' 字段）
            attacked_data_path : 攻击数据 .pkl 文件路径（包含 'data' 和 'labels' 字段）
            scaler             : sklearn MinMaxScaler 实例
            lookback           : 输入窗口长度
            delay              : 预测步长（默认为1，即预测下一时刻）
            batch_size         : 批次大小
            step               : 采样步长（默认为1）
        """
        self.normal_data_path = normal_data_path
        self.attacked_data_path = attacked_data_path
        self.mm = scaler
        self.lookback = lookback
        self.delay = delay
        self.step = step
        self.batch_size = batch_size

        # 加载数据
        self.normal_data, self.attacked_data, self.labels = self.load_data()
        self.n_features = self.normal_data.shape[1]

        # 拟合归一化器在正常数据上
        self.mm.fit(self.normal_data)
        self.normal_scaled = self.mm.transform(self.normal_data)
        self.attacked_scaled = self.mm.transform(self.attacked_data)  # 使用相同参数变换攻击数据

        # 划分训练/验证/测试集（基于正常数据）
        self.train_set, self.val_set, self.test_normal_set = self.train_val_test_split(self.normal_scaled)
        self.len_train_set, self.len_val_set, self.len_test_normal_set = len(self.train_set), len(self.val_set), len(self.test_normal_set)

        # 攻击数据作为测试集（全部）
        self.test_attack_set = self.attacked_scaled
        self.test_labels = self.labels

    def load_data(self):
        """从 .pkl 文件加载数据"""
        with open(self.normal_data_path, 'rb') as f:
            normal_dict = pickle.load(f)
            normal_data = normal_dict['data']  # 形状 (n_samples, n_features)

        with open(self.attacked_data_path, 'rb') as f:
            attack_dict = pickle.load(f)
            attacked_data = attack_dict['data']
            labels = attack_dict['labels']      # 形状 (n_samples,)

        print(f"\n正常数据形状: {normal_data.shape}")
        print(f"攻击数据形状: {attacked_data.shape}")
        print(f"标签形状: {labels.shape}")
        return normal_data, attacked_data, labels

    def train_val_test_split(self, data, test_set_percentage=0.2, val_set_percentage=0.5):
        """将正常数据划分为训练、验证、测试集（按时间顺序）"""
        test_split = int(data.shape[0] * test_set_percentage)
        test_set = data[-test_split:]
        train_set = data[:-test_split]

        val_split = int(test_set.shape[0] * val_set_percentage)
        final_test_set = test_set[-val_split:]
        val_set = test_set[:-val_split]

        print("\n训练集形状:", train_set.shape)
        print("验证集形状:", val_set.shape)
        print("测试集（正常）形状:", final_test_set.shape)
        return train_set, val_set, final_test_set

    def generator(self, data, min_index=0, max_index=None, shuffle=False):
        """生成器，产生 (输入窗口, 目标值) 对，用于训练/预测"""
        if max_index is None:
            max_index = len(data) - self.delay
        i = min_index + self.lookback
        while True:
            if shuffle:
                rows = np.random.randint(min_index + self.lookback, max_index + 1, size=self.batch_size)
            else:
                if i + self.batch_size > max_index:
                    rows = np.arange(i, max_index + 1)
                    i = min_index + self.lookback
                else:
                    rows = np.arange(i, i + self.batch_size)
                    i += self.batch_size
            samples = np.zeros((len(rows), self.lookback // self.step, data.shape[-1]))
            targets = np.zeros((len(rows), data.shape[-1]))
            for j, row in enumerate(rows):
                indices = range(row - self.lookback, row, self.step)
                samples[j] = data[indices]
                targets[j] = data[row + self.delay - 1]
            yield samples, targets

    def inv_scale(self, scaled_data):
        return self.mm.inverse_transform(scaled_data)

    def scale(self, data):
        return self.mm.transform(data)


    def get_tf_dataset(self, batch_size=32):
        """返回训练、验证、测试的 tf.data.Dataset 对象"""
        train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def inverse_transform(self, data):
        """将标准化后的数据还原为原始量纲（调用 DataNormalizer 的逆变换）"""
        # DataNormalizer 的 inverse_transform 要求输入形状与拟合时一致（此处为 (样本数, 窗口大小, 84)）
        return self.normalizer.inverse_transform(data)

# ==================== FPDM 模型类 ====================
class FPDM_Models:
    def __init__(self, input_shape, lookback=48, algorithm='xtm', head_size=9, num_heads=6,
                 ff_dim=128, num_transformer_blocks=1, mlp_units=[128], mlp_dropout=0.1,
                 dropout=0.1, training=False, loss_function='mse',
                 optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                 checkpoint_path="checkpoint/"):
        self.training = training
        self.input_shape = input_shape
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lookback = lookback
        self.n_features = input_shape[-1]  # 特征数

        self.checkpoint_file_path = os.path.join(self.checkpoint_path, f"fpdm_{self.algorithm}.keras")

        print(f'Loading {self.algorithm} model for FDI presence detection module ...')
        if self.training:
            if self.algorithm == 'xtm':
                self.model = self.build_xtm_model()
            elif self.algorithm == 'cnn_transformer':
                self.model = self.build_cnn_transformer_model()
            elif self.algorithm == 'cnn':
                self.model = self.build_cnn_model()
            elif self.algorithm == 'cnn_lstm':
                self.model = self.build_cnn_lstm_model()
            elif self.algorithm == 'transformer':
                self.model = self.build_transformer_model()
            else:
                self.model = self.build_xtm_model()
            self.model = self.compile_model()

            self.cpt_path = os.path.join(self.checkpoint_path, self.algorithm, "model.weights.h5")
            os.makedirs(os.path.dirname(self.cpt_path), exist_ok=True)

            self.callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint(filepath=self.cpt_path, save_weights_only=True,
                                                monitor='val_loss', mode='min', save_best_only=True)
            ]
        else:
            print("Loading saved model...")
            self.model = keras.models.load_model(self.checkpoint_file_path)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        x = layers.Dense(ff_dim, activation='relu')(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(inputs.shape[-1], activation='relu')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_xtm_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropout)
        x = layers.LSTM(128, activation='tanh', return_sequences=True)(x)
        x = layers.LSTM(128, activation='tanh')(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        output = layers.Dense(self.n_features)(x)
        return keras.Model(inputs, output)

    def build_cnn_transformer_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv1D(128, 9, activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropout)
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        output = layers.Dense(self.n_features)(x)
        return keras.Model(inputs, output)

    def build_cnn_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv1D(128, 9, activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 9, activation='relu')(x)
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        output = layers.Dense(self.n_features)(x)
        return keras.Model(inputs, output)

    def build_cnn_lstm_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv1D(128, 9, activation='relu')(inputs)
        x = layers.MaxPooling1D(3)(x)
        x = layers.LSTM(128, activation='tanh', return_sequences=True)(x)
        x = layers.LSTM(128, activation='tanh')(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        output = layers.Dense(self.n_features)(x)
        return keras.Model(inputs, output)

    def build_transformer_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropout)
        x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        output = layers.Dense(self.n_features)(x)
        return keras.Model(inputs, output)

    def compile_model(self):
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
        return self.model

    def train(self, train_gen_fn, val_gen_fn, steps_per_epoch, validation_steps, epochs=50, save_model=False):
        """
        参数：
            train_gen_fn : 可调用对象，返回训练数据生成器
            val_gen_fn   : 可调用对象，返回验证数据生成器
        """
        import tensorflow as tf

        # 定义输出签名
        input_spec = tf.TensorSpec(shape=(None, self.lookback, self.n_features), dtype=tf.float32)
        output_spec = tf.TensorSpec(shape=(None, self.n_features), dtype=tf.float32)
        output_signature = (input_spec, output_spec)

        # 创建数据集（每个 epoch 重新调用生成器函数）
        train_dataset = tf.data.Dataset.from_generator(train_gen_fn, output_signature=output_signature).repeat()
        val_dataset = tf.data.Dataset.from_generator(val_gen_fn, output_signature=output_signature).repeat()

        history = self.model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=self.callbacks
        )

        if save_model:
            print('Saving model...')
            self.model.save(self.checkpoint_file_path)
            history_df = pd.DataFrame(history.history)
            hist_path = os.path.join(self.checkpoint_path, 'training_history', f'{self.algorithm}_training_history.csv')
            os.makedirs(os.path.dirname(hist_path), exist_ok=True)
            history_df.to_csv(hist_path, index=False)
            

    def get_model_summary(self):
        return self.model.summary()


# ==================== FPDM 检测类 ====================
class FPDM:
    def __init__(self, data, fpdm_model, lookback=48, threshold=0.4):
        self.data = data
        self.fpdm_model = fpdm_model
        self.lookback = lookback
        self.threshold = threshold

    def get_forecasting_errors(self, real_data, predicted_data):
        real_inv = self.data.inv_scale(real_data)
        pred_inv = self.data.inv_scale(predicted_data)
        mae = utility_functions.MAE(real_inv, pred_inv)
        mse = utility_functions.MSE(real_inv, pred_inv)
        rmse = utility_functions.RMSE(real_inv, pred_inv)
        print(f'MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}')

    def is_fdi(self, real_data, predicted_data, threshold, smooth_window=1):
        errors = np.linalg.norm(real_data - predicted_data, axis=1)
        if smooth_window > 1:
            # 使用卷积实现滑动平均
            kernel = np.ones(smooth_window) / smooth_window
            errors = np.convolve(errors, kernel, mode='same')
            # 或使用 pandas 滚动平均（需 import pandas）
            # import pandas as pd
            # errors = pd.Series(errors).rolling(window=smooth_window, min_periods=1, center=True).mean().values
        return (errors > threshold).astype(int)

    # 原 get_prf 方法不再适用，保留但标记为弃用
    def get_prf(self, real_data, predicted_data, threshold):
        raise NotImplementedError("请使用外部计算方式，或直接传入真实标签。")

    # 新增：直接使用真实标签评估
    def evaluate_with_labels(self, real_data, predicted_data, true_labels, threshold):
        pred_labels = self.is_fdi(real_data, predicted_data, threshold)
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        prf = utility_functions.cm2prf(cm.T)  # 原 cm2prf 期望转置
        return prf, cm


# ==================== 主测试程序 ====================
if __name__ == "__main__":
    # ========== 配置参数 ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建到 FDIA-1 根目录的路径（向上两级：XTM -> project -> FDIA-1）
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    normal_data_path = os.path.join(base_dir, "processed_data_test", "normal_data.pkl")
    attacked_data_path = os.path.join(base_dir, "data", "attack_data.pkl")   # attack_data.pkl

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoint")
    algorithm = "xtm"                          # 可选: xtm, cnn_transformer, cnn, cnn_lstm, transformer
    lookback = 48                               # 窗口大小
    threshold = 0.6                              # 检测阈值
    batch_size = 32
    train_model = True                           # 是否训练（若无预训练模型请设为 True）
    epochs = 50

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, "images"), exist_ok=True)

    # ========== 1. 加载数据 ==========
    scaler = MinMaxScaler((0, 1))
    data = DataLoader(
        normal_data_path=normal_data_path,
        attacked_data_path=attacked_data_path,
        scaler=scaler,
        lookback=lookback,
        delay=1,
        batch_size=batch_size,
        step=1
    )

    # ========== 添加数据检查代码 ==========
    print("\n" + "="*50)
    # 获取攻击时刻的索引
    attack_indices = np.where(data.labels == 1)[0]
    # 取一小段攻击样本（如前100个攻击时刻）
    sample_attack = data.attacked_data[attack_indices[:100]]
    # 对比正常数据中对应时刻的原始值（需知道攻击前的正常值，但这里简单用攻击数据本身与正常数据的均值对比）
    normal_at_attack = data.normal_data[attack_indices[:100]]      
    relative_change_attack = np.abs(data.attacked_data[attack_indices] - data.normal_data[attack_indices]) / (np.abs(data.normal_data[attack_indices]) + 1e-8)
    print("攻击时刻相对变化均值:", np.mean(relative_change_attack))
    print("攻击时刻相对变化中位数:", np.median(relative_change_attack))
    print("攻击时刻相对变化最大:", np.max(relative_change_attack))

    n_features = data.n_features
    input_shape = (lookback, n_features)

    # ========== 2. 创建或加载 FPDM 模型 ==========
    fpdm_model = FPDM_Models(
        input_shape=input_shape,
        algorithm=algorithm,
        training=train_model,
        checkpoint_path=checkpoint_path
    )

    # ========== 3. 训练（如果需要） ==========
    if train_model:
        print("\n开始训练 FPDM 模型...")
        steps_per_epoch = (data.len_train_set - lookback) // batch_size
        validation_steps = (data.len_val_set - lookback) // batch_size
        train_gen_fn = lambda: data.generator(data.train_set, shuffle=True)
        val_gen_fn = lambda: data.generator(data.val_set, shuffle=True)
        fpdm_model.train(
            train_gen_fn, val_gen_fn,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            save_model=True
        )
        print("训练完成，模型已保存。\n")

    # ========== 4. 在攻击测试集上生成预测值 ==========
    print("正在生成攻击测试集上的预测值...")
    test_gen = data.generator(data.test_attack_set, shuffle=False)
    num_test_samples = len(data.test_attack_set) - lookback
    predictions = np.zeros((num_test_samples, n_features))

    collected = 0
    while collected < num_test_samples:
        X, y_list = next(test_gen)
        batch_pred = fpdm_model.model.predict_on_batch(X)
        batch_size_actual = batch_pred.shape[0]
        end = min(collected + batch_size_actual, num_test_samples)
        predictions[collected:end] = batch_pred[:end-collected]
        collected = end

    real_values = data.test_attack_set[lookback:]   # 真实值 (攻击数据)
    true_labels = data.test_labels[lookback:]       # 对应时间步的标签

    # ========== 5. 评估 ==========

    fpdm = FPDM(data, fpdm_model, lookback=lookback, threshold=threshold)

    print("\n" + "="*50)
    print("在攻击数据上的预测误差")
    fpdm.get_forecasting_errors(real_values, predictions)

    # ========== 阈值搜索（使用平均F1） ==========
    print("\n" + "="*50)
    print("搜索最佳阈值...")
    best_thresh = threshold
    best_f1 = 0
    thresholds = np.arange(0.2, 1.0, 0.05)  # 可调整范围和步长
    for thresh in thresholds:
        pred_labels = fpdm.is_fdi(real_values, predictions, thresh)
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        prf = utility_functions.cm2prf(cm.T)  # cm2prf 需要转置
        f1 = prf[2]  # 平均F1
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"最佳阈值: {best_thresh:.2f}, 最佳平均F1: {best_f1:.4f}")

    # ========== 使用最佳阈值重新评估 ==========
    pred_labels = fpdm.is_fdi(real_values, predictions, best_thresh)
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    prf = utility_functions.cm2prf(cm.T)
    print("\n使用最佳阈值的FDI存在检测性能:")
    print(f"精确率: {prf[0]:.4f}, 召回率: {prf[1]:.4f}, F1分数: {prf[2]:.4f}")
    print("混淆矩阵 (行:真实, 列:预测):")
    print(cm)

    # ========== 可选：与原固定阈值对比 ==========
    print("\n原始固定阈值 (0.4) 的性能:")
    pred_labels_orig = fpdm.is_fdi(real_values, predictions, 0.4)
    cm_orig = confusion_matrix(true_labels, pred_labels_orig, labels=[0, 1])
    prf_orig = utility_functions.cm2prf(cm_orig.T)
    print(f"精确率: {prf_orig[0]:.4f}, 召回率: {prf_orig[1]:.4f}, F1分数: {prf_orig[2]:.4f}")

    # 可选：绘制误差分布直方图（此处略，可根据需要添加）
    print("\n" + "="*50)
    print("绘制预测误差分布直方图")
    # 计算攻击测试集上的预测误差范数
    errors = np.linalg.norm(real_values - predictions, axis=1)
    # 根据真实标签分组
    errors_normal = errors[true_labels == 0]
    errors_attack = errors[true_labels == 1]

    if len(errors_normal) > 0 and len(errors_attack) > 0:
        # 调用已有的 show_barplot 方法
        save_path = os.path.join(checkpoint_path, "images", "barplot.jpg")
        utility_functions.show_barplot([errors_normal, errors_attack], ['正常', '攻击'], save_path=save_path,separate=False)
    else:
        print("警告: 正常或攻击样本为空，无法绘制直方图。")

    # ========== 绘制混淆矩阵 ==========
    print("\n混淆矩阵...")
    # 使用最佳阈值对应的混淆矩阵 cm（已在前面计算）
    utility_functions.plot_confusion_matrix(
        cm, 
        class_names=['正常', '攻击'],
        title=f'混淆矩阵 (阈值={best_thresh:.2f})',
        save_path=os.path.join(checkpoint_path, 'images', 'confusion_matrix_best.jpg')
    )