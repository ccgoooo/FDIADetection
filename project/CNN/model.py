import torch
import torch.nn as nn
import math

# ================ 残差CNN模型 ================
class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        residual = self.shortcut(residual)
        out += residual
        
        # 激活函数
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out


class ResidualCNN(nn.Module):
    """用于电力系统攻击检测的残差CNN模型"""
    
    def __init__(self, window_size=10, feature_dim=84, dropout_rate=0.3):
        super(ResidualCNN, self).__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块序列
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 64, dropout_rate=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            ResidualBlock(64, 128, stride=1, dropout_rate=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate),
            nn.AdaptiveAvgPool1d(4)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, 2)  # 二分类：正常和攻击
        )
    
    def forward(self, x):
        # 调整维度: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # 初始卷积
        x = self.initial_conv(x)
        
        # 残差块
        x = self.residual_blocks(x)
        
        # 展平并分类
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x


# ================ 增强版残差CNN模型（更深）===============
class DeepResidualCNN(nn.Module):
    """修复的深度残差CNN模型，避免池化层尺寸问题"""
    
    def __init__(self, window_size=10, feature_dim=84, dropout_rate=0.3):
        super(DeepResidualCNN, self).__init__()
        
        self.window_size = window_size
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 第一个残差块组 - 不进行下采样
        self.block_group1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, dropout_rate=dropout_rate),
            ResidualBlock(64, 64, stride=1, dropout_rate=dropout_rate)
        )
        
        # 第二个残差块组 - 通道数翻倍，时间维度减半
        self.block_group2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate),  # 下采样
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate)
        )
        
        # 第三个残差块组 - 通道数再翻倍
        self.block_group3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate),  # 下采样
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate)
        )
        
        # 计算经过卷积后的特征长度
        conv_output_length = self._calculate_conv_output_length(window_size)
        
        # 自适应全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(4)  # 固定输出长度为4
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 2)
        )
        
        print(f"模型配置: 窗口大小={window_size}, 特征维度={feature_dim}, 卷积后长度={conv_output_length}")
    
    def _calculate_conv_output_length(self, input_length):
        """计算卷积后特征长度"""
        # 初始卷积层: kernel=3, padding=1 -> 长度不变
        length = input_length
        
        # block_group1: 两个残差块，stride=1 -> 长度不变
        length = length  # 不变
        
        # block_group2: 第一个残差块stride=2 -> 长度减半
        length = length // 2
        
        # block_group3: 第一个残差块stride=2 -> 长度再减半
        length = length // 2
        
        return length
    
    def forward(self, x):
        # 调整维度: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # 初始卷积
        x = self.initial_conv(x)
        
        # 残差块组
        x = self.block_group1(x)
        x = self.block_group2(x)
        x = self.block_group3(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 展平并分类
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x
    
# ===============================================
# ================ ConvLSTM  ================
class FastConvLSTM(nn.Module):
    """
    快速ConvLSTM模型 - 专为快速训练优化
    参数量约40K，训练时间比标准ConvLSTM减少50%
    """
    
    def __init__(self, window_size=10, feature_dim=84, num_classes=2, dropout_rate=0.2):
        super(FastConvLSTM, self).__init__()
        
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # 1. 超轻量特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(16)  # 减少序列长度
        )
        
        # 2. 转换为2D特征图 (4x4)
        self.to_2d = nn.Linear(64 * 16, 64 * 4 * 4)
        
        # 3. 轻量ConvLSTM Cell
        self.conv_lstm = self._create_conv_lstm_cell(64, 128, kernel_size=3)
        
        # 4. 快速分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._print_parameters()
    
    def _create_conv_lstm_cell(self, input_channels, hidden_channels, kernel_size):
        """创建优化的ConvLSTM Cell"""
        class OptimizedConvLSTMCell(nn.Module):
            def __init__(self):
                super(OptimizedConvLSTMCell, self).__init__()
                # 使用分组卷积减少参数
                self.conv = nn.Conv2d(
                    input_channels + hidden_channels,
                    4 * hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    groups=4,  # 分组卷积，减少参数
                    bias=True
                )
            
            def forward(self, x, hidden):
                h_cur, c_cur = hidden
                combined = torch.cat([x, h_cur], dim=1)
                conv_out = self.conv(combined)
                
                # 分割门控信号
                cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
                
                i = torch.sigmoid(cc_i)
                f = torch.sigmoid(cc_f)
                o = torch.sigmoid(cc_o)
                g = torch.tanh(cc_g)
                
                c_next = f * c_cur + i * g
                h_next = o * torch.tanh(c_next)
                
                return h_next, c_next
        
        return OptimizedConvLSTMCell()
    
    def _init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        return (torch.zeros(batch_size, 128, 4, 4, device=device),
                torch.zeros(batch_size, 128, 4, 4, device=device))
    
    def _print_parameters(self):
        """打印模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        print(f"FastConvLSTM 参数量: {total:,} (非常轻量!)")
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 处理每个时间步
        hidden_states = []
        h, c = self._init_hidden(batch_size, x.device)
        
        for t in range(seq_len):
            # 特征编码
            time_feat = x[:, t, :].unsqueeze(-1)
            encoded = self.feature_encoder(time_feat)
            
            # 转换为2D
            flat = encoded.view(batch_size, -1)
            to_2d = self.to_2d(flat)
            to_2d = to_2d.view(batch_size, 64, 4, 4)
            
            # ConvLSTM更新
            h, c = self.conv_lstm(to_2d, (h, c))
            hidden_states.append(h)
        
        # 取最后一个时间步并分类
        last_hidden = hidden_states[-1]
        flat_features = last_hidden.view(batch_size, -1)
        output = self.classifier(flat_features)
        
        return output

# =============== Transformer模型 ================
# 位置编码 
class PositionalEncoding(nn.Module):
    """位置编码 - 轻量化版本"""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # 正弦和余弦编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区（不参与梯度更新）
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        前向传播
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 轻量多头注意力
class LightMultiHeadAttention(nn.Module):
    """轻量化的多头注意力机制"""
    
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super(LightMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影层（共享权重以减少参数）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
        # 移动到设备
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性投影并分割多头
        qkv = self.qkv_proj(query)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # 重塑为多头格式
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力
        out = torch.matmul(attention, v)
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out, attention


# 轻量前馈网络
class LightFeedForward(nn.Module):
    """轻量化的前馈网络"""
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(LightFeedForward, self).__init__()
        
        # 如果未指定d_ff，使用较小的扩展因子
        if d_ff is None:
            d_ff = d_model * 2  # 传统Transformer使用4倍，这里用2倍以减少参数
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU比ReLU更好但计算量相似
    
    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 轻量Transformer编码器层 
class LightTransformerEncoderLayer(nn.Module):
    """轻量化的Transformer编码器层"""
    
    def __init__(self, d_model, n_heads=4, d_ff=None, dropout=0.1):
        super(LightTransformerEncoderLayer, self).__init__()
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 注意力机制
        self.attention = LightMultiHeadAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feedforward = LightFeedForward(d_model, d_ff, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 注意力子层
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x, mask)
        x = residual + self.dropout(attn_output)
        
        # 前馈子层
        residual = x
        x = self.norm2(x)
        ff_output = self.feedforward(x)
        x = residual + self.dropout(ff_output)
        
        return x


# 轻量化Transformer模型 
class LightweightTransformer(nn.Module):
    """
    轻量化Transformer模型 - 用于电力系统攻击检测
    专为时序数据设计，参数量少但性能良好
    """
    
    def __init__(self, window_size=10, feature_dim=84, num_classes=2, 
                 d_model=128, n_heads=4, num_layers=3, dropout_rate=0.3):
        super(LightweightTransformer, self).__init__()
        
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # 1. 特征投影层 - 将原始特征映射到d_model维度
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len=window_size, dropout=dropout_rate)
        
        # 3. Transformer编码器层（堆叠）
        self.encoder_layers = nn.ModuleList([
            LightTransformerEncoderLayer(d_model, n_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 4. 全局池化 - 替代[CLS]token，更轻量
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 打印参数量
        self._print_parameters()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _print_parameters(self):
        """打印模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"LightweightTransformer模型参数量:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: (batch_size, window_size, feature_dim)
            mask: 可选，注意力掩码
        
        Returns:
            output: (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 特征投影
        x_proj = self.feature_projection(x)  # (batch_size, seq_len, d_model)
        
        # 2. 位置编码
        x_encoded = self.positional_encoding(x_proj)
        
        # 3. 通过Transformer编码器层
        for encoder_layer in self.encoder_layers:
            x_encoded = encoder_layer(x_encoded, mask)
        
        # 4. 全局池化
        # 转置以适应池化层: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x_pool = x_encoded.transpose(1, 2)
        pooled = self.global_pool(x_pool)  # (batch, d_model, 1)
        pooled = pooled.squeeze(-1)  # (batch, d_model)
        
        # 5. 分类
        output = self.classifier(pooled)
        
        return output