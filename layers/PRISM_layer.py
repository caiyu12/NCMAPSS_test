from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from scipy.stats import linregress
import torch.nn as nn
import torch
import torch.nn.functional as F

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

class FreCNNChannel(nn.Module):
    """
    A class that performs frequency-domain processing using CNN along the channel dimension.

    Args:
        feature_size (int): The number of channels (N).
        seq_len (int): The sequence length (T).
        kernel_size (int): The kernel size for the CNN. Default: 3

    Input:
        x (torch.Tensor): Input tensor of shape [B, N, T], where B is batch size, N is the number of channels,
                          and T is the sequence length.

    Output:
        torch.Tensor: Output tensor of shape [B, N, T] after frequency-domain CNN processing along the channel dimension.
    """
    def __init__(self, feature_size, seq_len, kernel_size=3):
        super(FreCNNChannel, self).__init__()
        self.feature_size = feature_size
        self.seq_len = seq_len
        self.kernel_size = kernel_size

        # CNN for processing frequency domain (real and imaginary parts)
        self.conv = nn.Conv2d(
            in_channels=2,  # 2 channels for real and imaginary parts
            out_channels=2,  # Output 2 channels for real and imaginary parts
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Padding to maintain dimensions
            bias=True
        )

    def forward(self, x):
        """
        Forward pass for frequency-domain channel processing using CNN.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, T]([B, C, S]).

        Returns:
            torch.Tensor: Output tensor of shape [B, N, T].
        """
        B, N, T = x.shape

        # Permute to [B, T, N] for FFT along channel dimension
        x = x.permute(0, 2, 1)  # [B, T, N]

        # Apply FFT along the channel dimension (N)
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, T, N//2 + 1]

        # Split real and imaginary parts
        x_real = x_fft.real  # [B, T, N//2 + 1]
        x_imag = x_fft.imag  # [B, T, N//2 + 1]

        # Stack real and imaginary parts as two channels: [B, 2, T, N//2 + 1]
        x_freq = torch.stack([x_real, x_imag], dim=1)

        # Apply CNN: Treat T and N//2 + 1 as spatial dimensions
        x_freq = self.conv(x_freq)  # [B, 2, T, N//2 + 1]

        # Split back into real and imaginary parts
        y_real = x_freq[:, 0, :, :]  # [B, T, N//2 + 1]
        y_imag = x_freq[:, 1, :, :]  # [B, T, N//2 + 1]

        # Combine into complex tensor
        y = torch.complex(y_real, y_imag)  # [B, T, N//2 + 1]

        # Apply IFFT along the channel dimension
        x = torch.fft.irfft(y, n=N, dim=2, norm="ortho")  # [B, T, N]

        # Permute back to [B, N, T]
        x = x.permute(0, 2, 1)  # [B, N, T]

        return x

class DynamicInput(nn.Module):
    """
        A class that dynamically creates and manages linear layers based on the input feature dimension.

        This is useful when the input feature size may vary during runtime, and you need to adaptively
        create new linear projection layers for different input sizes. The layers are registered
        using `nn.ModuleDict` to ensure proper device placement and persistence.

        Args:
            out_features (int): The number of output features for the dynamically created linear layer.

        Input:
            x (torch.Tensor): Input tensor of shape [B, S, C'] or [B, C']
        """
    def __init__(self, out_features):
        super(DynamicInput, self).__init__()
        self.out_features = out_features
        self.linears = nn.ModuleDict()

    def forward(self, x):
        in_features = x.shape[-1]
        key = str(in_features)
        if key not in self.linears:
            self.linears[key] = nn.Linear(in_features, self.out_features)
            self.linears[key].to(x.device)
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
    A class that computes a robust anchor point via sequence smoothing and performs normalization.

    This module calculates a smoothed version of the input sequence using an average pooling layer,
    extracts the last element of the smoothed sequence as the anchor point, and normalizes the input
    sequence based on this anchor. The anchor computation is detached from gradient calculations
    during backpropagation.

    Args:
        smoothing_kernel_size (int): The kernel size for the smoother. Default: 3

    Input:
        x (torch.Tensor): Input tensor of shape [B, S, C], where B is batch size,
                          S is the sequence length, and C is the number of features.

    Output:
        tuple: (
            x_norm (torch.Tensor): Normalized sequence of shape [B, S, C],
            anchor_out (torch.Tensor): Extracted anchor point of shape [B, 1, C]
        )
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