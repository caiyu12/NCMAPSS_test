import numpy as np
from scipy.stats import linregress
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
import pywt
from skimage.transform import resize


class FreMLPChannel(nn.Module):
    """
    A class that encapsulates FreMLP and MLP_channel functionality for frequency-domain processing
    along the channel dimension.

    Args:
        embed_size (int): The embedding size (D).
        feature_size (int): The number of channels (N).
        seq_len (int): The sequence length (T).
        scale (float): The scale for initializing parameters. Default: 0.02
        sparsity_threshold (float): The threshold for soft shrinkage. Default: 0.01

    Input:
        x (torch.Tensor): Input tensor of shape [B, N, T], where B is batch size, N is the number of channels,
                          and T is the sequence length.

    Output:
        torch.Tensor: Output tensor of shape [B, N, T] after frequency-domain processing along the channel dimension.
    """
    def __init__(self, embed_size, feature_size, seq_len, scale=0.02, sparsity_threshold=0.01):
        super(FreMLPChannel, self).__init__()
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.scale = scale
        self.sparsity_threshold = sparsity_threshold

        # Learnable embedding to map input to higher dimension
        self.embeddings = nn.Parameter(torch.randn(1, embed_size))

        # Learnable parameters for frequency-domain MLP
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.conv = nn.Conv2d(
            in_channels=embed_size,
            out_channels=embed_size,
            kernel_size=3,
            padding= 1,
            bias=True
        )

        # Projection layer to map back to original shape
        self.projection = nn.Linear(embed_size, 1)

    def FreMLP(self, B, T, N, x):
        """
        Frequency-domain MLP processing.

        Args:
            B (int): Batch size.
            T (int): Sequence length.
            N (int): Number of channels.
            x (torch.Tensor): FFT-transformed input of shape [B, T, N//2 + 1, embed_size] (complex).

        Returns:
            torch.Tensor: Processed complex tensor of the same shape.
        """
        x_real = x.real
        x_imag = x.imag

        o1_real = F.relu(
            torch.einsum('btjd,dd->btjd', x_real, self.r1) -
            torch.einsum('btjd,dd->btjd', x_imag, self.i1) +
            self.rb1
        )
        o1_imag = F.relu(
            torch.einsum('btjd,dd->btjd', x_imag, self.r1) +
            torch.einsum('btjd,dd->btjd', x_real, self.i1) +
            self.ib1
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        """
        Forward pass for frequency-domain channel processing.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, N, T].
        """
        B, N, T = x.shape

        # Embed input to [B, N, T, D]
        x = x.unsqueeze(3) * self.embeddings  # [B, N, T, 1] * [1, D] -> [B, N, T, D]

        # Permute for FFT along channel dimension
        x = x.permute(0, 2, 1, 3)  # [B, T, N, D]

        # Apply FFT
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, T, N//2 + 1, D]

        # Process in frequency domain
        y = self.FreMLP(B, T, N, x_fft)

        # Apply IFFT
        x = torch.fft.irfft(y, n=N, dim=2, norm="ortho")  # [B, T, N, D]

        # Permute back
        x = x.permute(0, 2, 1, 3)  # [B, N, T, D]

        # Project to original shape
        x = self.projection(x).squeeze(3)  # [B, N, T]

        return x

class DynamicInput(nn.Module):
    def __init__(self, out_features):
        super(DynamicInput, self).__init__()
        self.out_features = out_features
        # 使用 ModuleDict 来持有动态创建的层
        # 它能确保里面的所有模块都被正确注册
        self.linears = nn.ModuleDict()

    def forward(self, x):
        in_features = x.shape[-1]
        # ModuleDict 的键必须是字符串
        key = str(in_features)

        # 如果这个尺寸的线性层还没被创建
        if key not in self.linears:
            # 创建新的线性层并存入 ModuleDict
            self.linears[key] = nn.Linear(in_features, self.out_features)
            # !! 重要 !!
            # 将新创建的层移动到和输入数据相同的设备上
            self.linears[key].to(x.device)

            # --- 关于优化器的说明 ---
            # 这种在 forward 中动态添加新参数的方法，
            # 要求优化器也能动态地添加参数组(add_param_group)。
            # 这是一个高级用法，如果您的输入尺寸种类不多，
            # 更简单的做法是在 __init__ 中预先定义好所有可能的层。
            # 但如果必须动态创建，ModuleDict 是第一步。

        # 从 ModuleDict 中获取并使用正确的层
        return self.linears[key](x)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class UnifiedGMMGate:
    """
    一个统一的GMM门控，同时进行决策和可解释的诊断。
    """

    def __init__(self, weight_threshold=0.2, distance_threshold=2.5):
        self.gmm = GaussianMixture(n_components=2, random_state=0)
        self.scaler = StandardScaler()
        self.weight_threshold = weight_threshold
        self.distance_threshold = distance_threshold
        print(
            f"UnifiedGMMGate initialized with weight_threshold={weight_threshold}, distance_threshold={distance_threshold}")

    @staticmethod
    def _get_statistical_features(x_batch):
        # 提取(slope, R^2)特征，此函数与之前相同
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()
        batch_size, seq_len, num_channels = x_batch.shape
        all_features = np.zeros((batch_size, num_channels, 2))
        for i in range(batch_size):
            for j in range(num_channels):
                series = x_batch[i, :, j]
                if len(series) < 10: continue
                try:
                    slope, _, r_val, _, _ = linregress(np.arange(len(series)), series)
                    if np.isfinite(slope):
                        all_features[i, j, 0] = slope
                        all_features[i, j, 1] = r_val ** 2
                except Exception:
                    continue
        return np.mean(all_features, axis=0)

    def __call__(self, x_batch):
        """
        执行GMM分析，返回决策和详细的“主次群体”通道图。
        """
        features = self._get_statistical_features(x_batch)

        if np.all(features == 0):
            return 0, {"main_cluster": list(range(features.shape[0]))}

        features_scaled = self.scaler.fit_transform(features)

        self.gmm.fit(features_scaled)

        weights = self.gmm.weights_
        means = self.gmm.means_
        distance = np.linalg.norm(means[0] - means[1])

        # 决策逻辑
        is_unbalanced = min(weights) < self.weight_threshold
        is_distant = distance > self.distance_threshold
        gate_output = 1 if is_unbalanced and is_distant else 0

        # --- 新增部分：识别主次群体并映射通道 ---
        # 预测每个通道的标签
        channel_labels = self.gmm.predict(features_scaled)

        # 找到权重较小的那个簇的ID，它就是“离群簇”
        outlier_cluster_id = np.argmin(weights)

        cluster_map = defaultdict(list)
        for channel_idx, cluster_id in enumerate(channel_labels):
            if cluster_id == outlier_cluster_id and gate_output == 1:
                # 只有在最终决策为1（异构）时，才真正标记为离群簇
                cluster_map["outlier_cluster"].append(channel_idx)
            else:
                # 其他情况（包括决策为0时），都归为主群体
                cluster_map["main_cluster"].append(channel_idx)

        return gate_output, dict(cluster_map)

class InterpretableClusteringGate:
    """
    一个可解释的、基于DBSCAN聚类的门控。
    它不仅做出决策，还能返回每个簇包含的通道编号。
    """

    def __init__(self, eps=0.5, min_samples=2):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler = StandardScaler()
        print(f"InterpretableClusteringGate initialized with DBSCAN(eps={eps}, min_samples={min_samples})")

    @staticmethod
    def _get_statistical_features(x_batch):
        # (此静态方法与之前版本完全相同)
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()

        batch_size, seq_len, num_channels = x_batch.shape
        all_features = np.zeros((batch_size, num_channels, 2))

        for i in range(batch_size):
            for j in range(num_channels):
                series = x_batch[i, :, j]
                if len(series) < 10: continue
                try:
                    slope, _, r_val, _, _ = linregress(np.arange(len(series)), series)
                    if np.isfinite(slope):
                        all_features[i, j, 0] = slope
                        all_features[i, j, 1] = r_val ** 2
                except Exception:
                    continue

        return np.mean(all_features, axis=0)

    def __call__(self, x_batch):
        """
        执行聚类分析，并返回决策和详细的通道聚类图。

        返回:
        tuple: (门控信号 (0或1), 每个簇及其包含的通道编号字典)
        """
        features = self._get_statistical_features(x_batch)

        if np.all(features == 0):
            # 如果所有特征都为0，所有通道都属于簇0
            all_channels = list(range(features.shape[0]))
            return 0, {0: all_channels}

        features_scaled = self.scaler.fit_transform(features)

        self.dbscan.fit(features_scaled)
        labels = self.dbscan.labels_

        # --- 新增部分：构建簇与通道的映射 ---
        cluster_map = defaultdict(list)
        for channel_idx, cluster_id in enumerate(labels):
            cluster_map[int(cluster_id)].append(channel_idx)

        # 将defaultdict转换为普通dict以便打印
        cluster_map = dict(cluster_map)

        # 决策逻辑保持不变
        gate_output = 1 if len(set(labels)) > 1 else 0

        return gate_output, cluster_map

class SmoothedAnchorNorm(nn.Module):
    """
    通过平滑序列来获取鲁棒锚点，并进行归一化。反向传播时锚点的计算过程不参与梯度更新。
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

class Cluster_RNN_improved(nn.Module):
    def __init__(self, configs):
        super(Cluster_RNN_improved, self).__init__()
        self.name = 'Cluster_RNN'
        self.layer = configs.e_layers
        self.accept_window = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.seq_len = configs.seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.pred_len = 1
        self.dropout = configs.dropout
        self.lstm_layer_num = 1
        self.register_buffer('gate_value', None)
        self.cluster_map = None
        self.batch_size = None
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, self.d_model))
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.valueSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.seg_len),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )
        # self.SensorsEmbedding = nn.Sequential(
        #     DynamicInput(self.d_model),
        #     nn.ReLU()
        # )
        self.SensorsEmbedding = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU()
        )
        self.SensorSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )
        self.dynamic_squeeze_layers = nn.ModuleDict()
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.norm = nn.BatchNorm1d(self.seq_len)
        self.projection = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze = nn.Sequential(
            nn.Linear(self.enc_in, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.robust_norm = SmoothedAnchorNorm()
        self.cluster_map_generator = InterpretableClusteringGate()
        self.cluster_gate = UnifiedGMMGate()
        self.DBSCAN_gate = InterpretableClusteringGate()
        self.rnn_cd = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num,
                          batch_first=True)
        self.rnn_ci = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num,
                          batch_first=True)
        self.channel = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.d_model, self.enc_in),
            # nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.seg_num_x, self.seq_len)
        )
        self.FreMLP = FreMLPChannel(embed_size=32, feature_size=self.enc_in, seq_len=self.seq_len)
        self.SeriesDecomp = series_decomp(kernel_size=5)
    def forecast_cd_shifted(self, x_enc, channel_size):
        """
        使用通道依赖策略的RNN，将长序列分为多个子序列并行处理，
        由此减轻因序列过长导致的RNN遗忘问题。
        input:b,s,c; output:b,s,c
        """
        batch_size, _, channels = x_enc.shape
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, channel_size).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x = self.rnn_cd(x)[0]  # bs,n,d
        key = str(channel_size)
        if key not in self.dynamic_squeeze_layers:
            self.dynamic_squeeze_layers[key] = nn.Linear(self.d_model, channel_size).to(x_enc.device)
        x = self.dynamic_squeeze_layers[key](x)
        x = F.relu(x).reshape(batch_size, -1, channel_size)  # 手动应用ReLU
        # self.SensorSqueeze[0] = nn.Linear(self.d_model, channels).to(x_enc.device)
        # x = self.SensorSqueeze(x).reshape(batch_size, -1, channel_size)
        x = self.norm(x)  # b,s,c
        return x

    def forecast_cd(self, x_enc):
        """
        使用通道依赖策略的RNN，将长序列分为多个子序列并行处理，
        由此减轻因序列过长导致的RNN遗忘问题。
        input:b,s,c; output:b,s,c
        """
        batch_size, _, _ = x_enc.shape
        x_enc = self.FreMLP(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x, hn = self.rnn_cd(x) #x:bs,n,d  hn:1,bs,d
        x = self.SensorSqueeze(x).reshape(batch_size, -1, self.enc_in) #b,s,c
        x = self.norm(x) #b,s,c
        return x

    def forecast_ci(self, x_enc):
        """
        使用通道独立策略的RNN，将通道维度并到batch维度从而实现并行训练。
        input:b,s,c; output:b,s,c
        """
        batch_size, _, _ = x_enc.shape
        # x_enc, seq_last = self.robust_norm(x_enc)
        x = x_enc.permute(0, 2, 1)  # b,c,s
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len)) #bc,n,d
        x, hn = self.rnn_ci(x) #bc,n,d  1,bc,d
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.d_model) #b,c,n,d
        x = self.valueSqueeze(x).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1) #b,w,c
        x = self.channel(x)
        return x

    # def forward(self, x_enc):
    #     batch_size = x_enc.shape[0]  # b,w,c
    #     if self.gate_value is None or self.gate_value.device != x_enc.device:
    #         gate, cluster_map = self.cluster_gate(x_enc)
    #         self.gate_value = torch.tensor([gate], device=x_enc.device)
    #         self.cluster_map = cluster_map
    #     else:
    #         gate = self.gate_value.item()
    #     if gate == 0:
    #         x = self.forecast_cd(x_enc,self.enc_in)
    #     else:
    #         x = self.forecast(x_enc)
    #     enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
    #     enc_out_2d = enc_out.view(-1, self.enc_in)
    #     enc_output = self.squeeze(enc_out_2d)
    #     return enc_output

    def forecast(self, x_enc):
        batch_size = x_enc.shape[0]
        x_enc, seq_last = self.robust_norm(x_enc)
        # 处理各聚类组（全部使用 forecast_cd）
        output_parts = []
        for cluster_id, channels in self.cluster_map.items():
            x_group = x_enc[:, :, channels]
            out_group = self.forecast_cd(x_group, len(channels))  # 所有簇均使用 CD
            output_parts.append((channels, out_group))

        # 构建通道到输出的映射
        channel_to_output = {}
        for channels, out_group in output_parts:
            for i, channel in enumerate(channels):
                channel_to_output[channel] = out_group[:, :, i:i + 1]

        # 按原始通道顺序拼接
        sorted_channels = sorted(channel_to_output.keys())
        output_list = [channel_to_output[c] for c in sorted_channels]
        output = torch.cat(output_list, dim=-1)
        final_output = self.forecast_ci(output) + seq_last
        return

    def forward(self, x_enc):
        # x_enc: [b, s, c]
        batch_size = x_enc.shape[0]
        self.batch_size = batch_size
        if self.gate_value is None or self.gate_value.device != x_enc.device:
            gate, cluster_map = self.cluster_gate(x_enc)
            # cluster_map_ori = self.cluster_map_generator(x_enc)[1]
            # cluster_map = {
            #     "main_cluster": cluster_map_ori.get(0, []),
            #     "outlier_cluster": [
            #         channel
            #         for key in cluster_map_ori
            #         if key != 0
            #         for channel in cluster_map_ori[key]
            #     ]
            # }
            self.gate_value = torch.tensor([gate], device=x_enc.device)
            self.cluster_map = cluster_map  # 缓存cluster_map以供使用
        else:
            gate = self.gate_value.item()
            cluster_map = self.cluster_map


        # 2. 根据gate值选择宏观处理流程
        if gate == 0:
            x = self.forecast_cd(x_enc)
        else:
            processed_x = torch.zeros_like(x_enc)
            seasonal_x = torch.zeros_like(x_enc)
            # 识别主群体和离群群体
            main_channels = cluster_map.get("main_cluster", [])
            outlier_channels = cluster_map.get("outlier_cluster", [])
            x_enc, seq_last = self.robust_norm(x_enc)
            # 对主群体进行增强 (使用通道依赖路径)
            x_outlier_group = x_enc[:, :, outlier_channels]
            outlier_seasonal, outlier_trend = self.SeriesDecomp(x_outlier_group)
            seasonal_x[:, :, outlier_channels] = outlier_seasonal
            seasonal_x[:, :, main_channels] = x_enc[:, :, main_channels]
            seasonal_output = self.forecast_cd(seasonal_x)
            processed_x[:, :, outlier_channels] = outlier_trend
            # seasonal_output = self.forecast_ci(seasonal_output) + seq_last
            # trend_output = self.forecast_ci(processed_x) + seq_last
            # x = trend_output + seasonal_output
            x = seasonal_output + processed_x
            x = self.forecast_ci(x) + seq_last
            # if main_channels:
            #     x_main_group = x_enc[:, :, main_channels]
            #     processed_main_group = self.forecast_cd_shifted(x_main_group, len(main_channels))
            #     processed_x[:, :, main_channels] = processed_main_group
            # if outlier_channels:
            #     processed_x[:, :, outlier_channels] = x_enc[:, :, outlier_channels]
            # x = self.forecast_ci(processed_x) + seq_last

        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.enc_in)
        enc_output = self.squeeze(enc_out_2d)

        return enc_output






