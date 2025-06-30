import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau
from itertools import combinations
from scipy.stats import linregress
from layers.RevIN import RevIN

class APRULHead(nn.Module):
    def __init__(self, seq_len, enc_in, d_model=64, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.feature_projection = nn.Linear(enc_in, d_model)
        self.temporal_attention = nn.Linear(d_model, 1)
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),  # 压缩到较小维度
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),  # 扩展回 d_model
            nn.Sigmoid()  # 生成通道权重
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.rul_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # 输入 x: [batch_size, seq_len, enc_in]
        x = self.feature_projection(x)  # [batch_size, seq_len, d_model]
        temporal_scores = self.temporal_attention(x)  # [batch_size, seq_len, 1]
        temporal_weights = F.softmax(temporal_scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(x * temporal_weights, dim=1)  # [batch_size, d_model]
        channel_weights = self.channel_attention(context_vector)  # [batch_size, d_model]
        context_vector = context_vector * channel_weights  # [batch_size, d_model]
        context_vector = self.dropout(context_vector)

        rul_output = self.rul_projection(context_vector)  # [batch_size, 1]
        return rul_output

class ResBlock(nn.Module):
    def __init__(self, sensors, seq_len, t_model, c_model, dropout):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(t_model, seq_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(sensors, c_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(c_model, sensors),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)
        self.channel_conv  = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)

        self.norm = nn.BatchNorm1d(seq_len)
    def forward(self, x):
        # x: [B, L, D]
        x_tprl = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x_chnl = self.channel(x)
        # x_aton = self.attention_layer(x)


        x_out = x + self.temporal_conv(x_tprl.unsqueeze(1)).squeeze(1) + self.channel_conv(x_chnl.unsqueeze(1)).squeeze(1)
        # x_out = x + x_tprl+ x_chnl
        x_out = self.norm(x_out)
        return x_out

class SmoothedAnchorNorm(nn.Module):
    """
    通过平滑序列来获取鲁棒锚点，并进行归一化。
    在反向传播时，锚点的计算过程不参与梯度更新。
    input: b,s,c; output:b,s,c  b,1,c
    """

    def __init__(self, smoothing_kernel_size=3):
        super(SmoothedAnchorNorm, self).__init__()
        self.smoother = nn.AvgPool1d(
            kernel_size=smoothing_kernel_size,
            stride=1,
            padding=(smoothing_kernel_size - 1) // 2
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        x_permuted = x.permute(0, 2, 1)  # b,c,s

        with torch.no_grad():
            x_smoothed = self.smoother(x_permuted)  # b,c,s
        # anchor: [batch, features, 1]
        anchor = x_smoothed[:, :, -1].unsqueeze(-1)  # b,c,1

        # x_norm_permuted: [batch, features, seq_len]
        x_norm_permuted = x_permuted - anchor

        x_norm = x_norm_permuted.permute(0, 2, 1)  # b,s,c
        anchor_out = anchor.permute(0, 2, 1)  # b,1,c

        return x_norm, anchor_out

class StatisticalTrendGate:
    """
    一个用于分析数据批次通道间趋势差异的门控类。
    该类通过统计方法判断一个批次的数据中，是否存在少数通道
    相较于其他通道表现出显著更强的趋势特性。
    """

    def __init__(self, p_threshold=0.05, cv_threshold=1.5):
        self.p_threshold = p_threshold
        self.cv_threshold = cv_threshold
        print(
            f"StatisticalTrendGate initialized with: p_threshold={self.p_threshold}, cv_threshold={self.cv_threshold}")

    def _sens_slope(self, x):
        # 私有辅助方法：计算一维数组的森斜率
        x = np.asarray(x)
        if x.ndim != 1 or len(x) < 2: return np.nan
        indices = list(combinations(range(len(x)), 2))
        slopes = [(x[j] - x[i]) / (j - i) for i, j in indices]
        return np.median(slopes)

    def __call__(self, x_batch):
        """
        使类的实例可调用，执行趋势分析并返回决策。

        参数:
        x_batch (np.array or torch.Tensor): 形状为 [b, s, c] 的输入数据批次。

        返回:
        tuple[int, float]: (门控信号 (0或1), 该批次的平均CV值)
        """
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()

        batch_size, seq_len, num_channels = x_batch.shape
        batch_cv_scores = []

        for i in range(batch_size):
            channel_trend_scores = []
            for j in range(num_channels):
                series = x_batch[i, :, j]
                _, p_value = kendalltau(np.arange(len(series)), series)
                if p_value < self.p_threshold:
                    slope = self._sens_slope(series)
                    channel_trend_scores.append(np.abs(slope))
                else:
                    channel_trend_scores.append(0)
            scores = np.array(channel_trend_scores)
            mean_score, std_dev = np.mean(scores), np.std(scores)

            if mean_score > 1e-9:
                cv = std_dev / mean_score
                batch_cv_scores.append(cv)

        if not batch_cv_scores:
            return 0, 0.0

        mean_batch_cv = np.mean(batch_cv_scores)
        gate_output = 1 if mean_batch_cv > self.cv_threshold else 0

        return gate_output, mean_batch_cv

class SequentialLightweightGate:
    """
    一个纯串行、不依赖任何并行库的轻量化趋势分析门控。

    它使用O(n)的线性回归作为核心，并在单个CPU核心上顺序执行。
    """

    def __init__(self, cv_threshold=1.5):
        self.cv_threshold = cv_threshold
        print(f"SequentialLightweightGate initialized with: cv_threshold={self.cv_threshold}")
        print("Running in pure sequential mode (no parallelization).")

    @staticmethod
    def _analyze_series_linear(series):
        """
        使用线性回归分析单个序列并返回其趋势分数。
        """
        series = np.asarray(series)
        seq_len = len(series)
        if seq_len < 10:
            return 0.0

        time_index = np.arange(seq_len)

        try:
            slope, _, r_value, _, _ = linregress(time_index, series)

            if not np.isfinite(slope):
                return 0.0

            r_squared = r_value ** 2
            score = np.abs(slope) * r_squared
            return score

        except Exception:
            return 0.0

    def __call__(self, x_batch):
        """
        执行串行的趋势分析并返回决策。
        """
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()

        batch_size, seq_len, num_channels = x_batch.shape

        # 使用简单的列表推导进行串行计算，取代之前的并行库
        trend_scores_flat = [
            self._analyze_series_linear(x_batch[i, :, j])
            for i in range(batch_size)
            for j in range(num_channels)
        ]

        trend_scores = np.array(trend_scores_flat).reshape(batch_size, num_channels)

        # 计算每个样本的CV值 (这部分已经是高效的NumPy操作)
        mean_scores = np.mean(trend_scores, axis=1)
        std_devs = np.std(trend_scores, axis=1)

        batch_cv_scores = np.divide(std_devs, mean_scores, out=np.zeros_like(mean_scores), where=mean_scores > 1e-9)

        mean_batch_cv = np.mean(batch_cv_scores)
        gate_output = 1 if mean_batch_cv > self.cv_threshold else 0

        return gate_output, mean_batch_cv

class FastTrendGate:
    """
    一个高效的趋势分析门控类，用于快速检测多维序列中是否存在具有较强趋势特征的通道。
    该类使用一阶差分和 L1 范数来近似趋势强度，并通过变异系数（CV）判断通道间的趋势差异。
    """

    def __init__(self, cv_threshold=1.5):
        """
        初始化门控。

        参数:
        cv_threshold (float): CV 阈值，用于判断通道间趋势分数差异性，默认值为 1.5。
        """
        self.cv_threshold = cv_threshold
        print(f"FastTrendGate initialized with: cv_threshold={self.cv_threshold}")

    def __call__(self, x_batch):
        """
        执行趋势分析并返回决策。

        参数:
        x_batch (np.array or torch.Tensor): 形状为 [batch_size, seq_len, num_channels] 的输入数据批次。

        返回:
        tuple[int, float]: (门控信号 (0 或 1), 该批次的平均 CV 值)
        """
        # 如果输入是 PyTorch Tensor，则转换为 NumPy 数组
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()

        batch_size, seq_len, num_channels = x_batch.shape

        # 处理序列长度小于 2 的情况
        if seq_len < 2:
            return 0, 0.0

        # 计算一阶差分，沿时间轴 (axis=1)
        diff = np.diff(x_batch, axis=1)  # 形状: [batch_size, seq_len-1, num_channels]

        # 计算差分序列的 L1 范数作为趋势强度
        trend_scores = np.sum(np.abs(diff), axis=1)  # 形状: [batch_size, num_channels]

        # 计算每个样本的通道趋势强度的均值和标准差
        mean_scores = np.mean(trend_scores, axis=1)  # 形状: [batch_size]
        std_devs = np.std(trend_scores, axis=1)  # 形状: [batch_size]

        # 计算 CV，防止除以零
        cv = np.divide(std_devs, mean_scores, out=np.zeros_like(mean_scores), where=mean_scores > 1e-9)

        # 计算批次的平均 CV
        mean_batch_cv = np.mean(cv)

        # 门控决策
        gate_output = 1 if mean_batch_cv > self.cv_threshold else 0

        return gate_output, mean_batch_cv

class EndToEndGate(nn.Module):
    """一个端到端在线学习的门控网络"""
    def __init__(self, num_channels, feature_dim=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(feature_dim)
        )
        self.decision_head = nn.Sequential(
            nn.Linear(num_channels * feature_dim*4, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 输出对应两条路径的logits
        )

    def forward(self, x_batch):
        b, s, c = x_batch.shape
        x_reshaped = x_batch.permute(0, 2, 1).reshape(b * c, 1, s)
        channel_features = self.feature_extractor(x_reshaped)
        features_flat = channel_features.reshape(b, -1)
        logits = self.decision_head(features_flat)
        return logits


class HybridStatisticalGate(nn.Module):
    """
    一个混合门控：先用统计方法提取趋势分数，再用神经网络进行端到端学习决策。
    """

    def __init__(self, num_channels, num_blocks=4):
        """
        初始化。

        参数:
        num_channels (int): 输入数据的通道数 (c)。
        num_blocks (int): 统计分析时使用的分块数。
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        # 可学习的决策头：一个简单的MLP，输入为c个通道的趋势分数
        self.decision_head = nn.Sequential(
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出对应两条路径的2个logits
        )

    @staticmethod
    def _get_trend_scores(x_batch, num_blocks):
        """
        [不可训练部分] 使用超轻量化的分块聚合分析提取趋势分数。
        """
        if isinstance(x_batch, torch.Tensor):
            device = x_batch.device
            x_batch = x_batch.detach().cpu().numpy()
        else:
            device = torch.device("cpu")

        batch_size, seq_len, num_channels = x_batch.shape

        def _analyze(series):
            if len(series) < num_blocks * 2: return 0.0
            try:
                blocks = np.array_split(series, num_blocks)
                block_means = [np.mean(b) for b in blocks]
                slope, _, r_val, _, _ = linregress(np.arange(num_blocks), block_means)
                return np.abs(slope) * (r_val ** 2) if np.isfinite(slope) else 0.0
            except Exception:
                return 0.0

        scores_flat = [_analyze(x_batch[i, :, j]) for i in range(batch_size) for j in range(num_channels)]
        scores_tensor = torch.tensor(scores_flat, dtype=torch.float32).view(batch_size, num_channels)
        return scores_tensor.to(device)

    def forward(self, x_batch):
        # 步骤1: 提取统计特征 [b, s, c] -> [b, c]
        trend_scores = self._get_trend_scores(x_batch, self.num_blocks)

        # 步骤2: 可学习的决策 [b, c] -> [b, 2]
        logits = self.decision_head(trend_scores)

        return logits

class CID_RNN(nn.Module):
    def __init__(self, configs):
        super(CID_RNN, self).__init__()
        self.name = 'LSTM_pTSMixer_GA'
        self.layer = configs.e_layers
        self.accept_window = configs.seq_len
        self.sensors = configs.enc_in
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.seq_len = configs.seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.dropout = configs.dropout
        self.lstm_layer_num = 1
        self.register_buffer('gate_value', None)
        # self.lstm = nn.LSTM(input_size=self.sensors, hidden_size=self.sensors, num_layers=self.lstm_layer_num, batch_first=True)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num, batch_first=True)
        self.rnn_ci = nn.GRU(input_size=self.seg_len, hidden_size=self.seg_len, num_layers=self.lstm_layer_num,
                          batch_first=True)
        self.ci_rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num,
                          batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.valueSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.seg_len),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )
        self.SensorsEmbedding = nn.Sequential(
            nn.Linear(self.sensors, self.d_model),
            nn.ReLU()
        )
        self.SensorSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )

        self.model = nn.ModuleList(
            [ResBlock(self.sensors, self.seq_len, self.d_model, self.d_model, self.dropout)
             for _ in range(configs.d_layers)]
        )
        self.norm = nn.BatchNorm1d(self.seq_len)

        self.pred_len = 1
        # self.projection = nn.Linear(seq_len, pred_len)
        # self.squeeze = nn.Linear(sensors, pred_len)
        self.projection = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze    = nn.Sequential(
            nn.Linear(self.sensors, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.SENet = APRULHead(self.seq_len, self.sensors, self.d_model, self.dropout)
        self.robust_norm = SmoothedAnchorNorm()
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)
        self.gate = nn.Parameter(torch.rand(1))
        self.gating_mlp = nn.Sequential(
            nn.Linear(self.sensors * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.stat_gate = SequentialLightweightGate()
        self.trend_gate = HybridStatisticalGate(self.enc_in)
        self.tau = 1.0

    def forecast(self, x_enc): #b,s,c
        batch_size = x_enc.shape[0]
        #simple norm
        # seq_last = x_enc[:, -1:, :].detach()
        # x_enc = (x_enc - seq_last)
        #archor norm
        x_enc, seq_last = self.robust_norm(x_enc)
        #std norm
        # x_enc = self.revin_layer(x_enc, 'norm')

        x_ci = x_enc.permute(0, 2, 1) #b,c,s
        x_ci = x_ci.reshape(batch_size, self.enc_in, self.seg_num_x, self.seg_len) #b,c,n,s
        x_ci = x_ci.reshape(batch_size*self.enc_in, self.seg_num_x, self.seg_len) #bc,n,s
        output_ci, hn_ci = self.rnn_ci(x_ci) #bc,n,s  1,bc,s
        hn_ci = hn_ci.reshape(batch_size, self.enc_in, self.seg_len).permute(0, 2, 1) #b,s,c
        hn_ci = self.SensorsEmbedding(hn_ci) #b,s,d
        hn_ci = hn_ci.reshape(1,-1,self.d_model) #1,bs,d
        output_ci = output_ci.reshape(batch_size, self.enc_in, self.seg_num_x, self.seg_len) #b,c,n,s
        output_ci = self.SensorsEmbedding(output_ci.permute(0, 2, 3, 1)).permute(0,2,1,3) #b,s,n,d
        output_ci = output_ci.reshape(batch_size*self.seg_len, self.seg_num_x, self.d_model) #bs,n,d

        #channel dependence
        # x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3)
        # x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        # x,hs = self.rnn(x) #bs,n,d   1,bs,d
        # output1 = self.rnn(output_ci, hs)[0]
        # output2 = self.rnn(x, hn_ci)[0]
        x = self.SensorSqueeze(output_ci).reshape(batch_size, -1, self.sensors)
        # x = self.SensorSqueeze(x).reshape(batch_size, -1, self.sensors)
        x = self.norm(x)

        #denorm
        # x = self.revin_layer(x, 'denorm')
        x = x + seq_last

        #rul-prediction-head
        # enc_output = self.SENet(x)
        # enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        # enc_out_2d = enc_out.view(-1, self.sensors)
        # enc_output = self.squeeze(enc_out_2d)
        return x

    def RUL_prediction_ci(self, x_enc):
        batch_size = x_enc.shape[0]
        x_enc, seq_last = self.robust_norm(x_enc)
        x = x_enc.permute(0, 2, 1) #b,c,w
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.seg_len).reshape(batch_size*self.enc_in, self.seg_num_x, self.seg_len) #b,c,n,s -> bc,n,s
        x_out, hn = self.ci_rnn(self.valueEmbedding(x)) #x_out: bc,n,d, hn: 1,bc,d
        x_out = self.valueSqueeze(x_out) #bc,n,s
        hn = self.valueSqueeze(hn) #1,bc,s
        x_out = x_out.reshape(batch_size, self.enc_in, self.seg_num_x, self.seg_len).permute(0, 2, 3, 1).reshape(batch_size, -1, self.enc_in) + seq_last #b,w,c
        return x_out, hn
    def RUL_prediction_cd(self, x_enc):
        batch_size = x_enc.shape[0]
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x, hs = self.rnn(x)  # bs,n,d   1,bs,d
        x = self.SensorSqueeze(x) # bs,n,c
        x = x.reshape(batch_size, self.seg_len, self.seg_num_x, self.sensors).permute(0, 2, 1, 3).reshape(batch_size, -1, self.sensors) #b,w,c
        hs = self.SensorSqueeze(hs) # 1,bs,c
        return x, hs


    def forward(self, x_enc):
        batch_size =  x_enc.shape[0] #b,w,c
        gate_logits = self.trend_gate(x_enc)  # [b, 2]

        # 2. 应用Gumbel-Softmax生成选择权重
        # hard=False在训练时使用软选择，保证梯度流
        # hard=True在推理时使用硬选择(one-hot)，更高效
        gate_weights = F.gumbel_softmax(gate_logits, tau=self.tau, hard=not self.training)

        w_ci = gate_weights[:, 0].view(-1, 1, 1)
        w_cd = gate_weights[:, 1].view(-1, 1, 1)
        #encode(channel independence and dependence)
        # x_enc_ci, hn_ci = self.RUL_prediction_ci(x_enc) #b,w,c
        # if self.gate_value is None or self.gate_value.device != x_enc.device:
        #     gate, cv = self.stat_gate(x_enc)
        #     self.gate_value = torch.tensor([gate], device=x_enc.device)
        # else:
        #     gate = self.gate_value.item()
        x_enc_ci = self.forecast(x_enc) #b,w,c
        x_enc_cd, hs_cd = self.RUL_prediction_cd(x_enc) #b,w,c
        # gate = torch.sigmoid(self.gating_mlp(gate_input)).unsqueeze(-1)  # [b, 1]
        # gate,cv = self.stat_gate(x_enc)
        print(w_ci)
        # x = x_enc_ci*gate + x_enc_cd*(1-gate)
        x = w_ci * x_enc_ci + w_cd * x_enc_cd
        # x = x_enc_cd*x_enc_ci
        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.sensors)
        enc_output = self.squeeze(enc_out_2d)
        # enc_output = self.forecast(x_enc)
        return enc_output[:, -self.pred_len:]  #

