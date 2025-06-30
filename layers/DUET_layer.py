import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.functional import gumbel_softmax, threshold
from einops import rearrange

class BaseMaskGenerator(nn.Module):
    """所有掩码生成器的基类，用于统一接口。"""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: [B, C, D]
        # 返回一个形状为 [B, C, C] 的关系矩阵（掩码）
        raise NotImplementedError

class LowRankMask(BaseMaskGenerator):
    """通过学习低秩因子来生成掩码。"""
    def __init__(self, d_model, rank=16):
        super().__init__(d_model)
        # 学习两个低秩的因子矩阵
        self.factor1 = nn.Linear(d_model, rank)
        self.factor2 = nn.Linear(d_model, rank)

    def forward(self, x):
        # x: [B, C, D]
        # f1, f2: [B, C, rank]
        f1 = self.factor1(x)
        f2 = self.factor2(x)
        # mask: [B, C, C]
        mask = torch.bmm(f1, f2.transpose(1, 2))
        return mask

class RevIN(nn.Module):
    """可逆实例归一化层."""

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        # x: [B, L, C]
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


class LinearExtractor(nn.Module):
    """使用可学习的基向量提取时间特征."""

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x_permuted = x.permute(0, 2, 1)
        # weights: [B, C, d_model] -> [B, d_model, C]
        weights = self.projection(x_permuted)
        return weights.transpose(1, 2)


class MahalanobisMask(nn.Module):
    """通过马氏距离生成注意力掩码."""

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        if C <= 1: return None
        x_mean = x - torch.mean(x, dim=2, keepdim=True)
        cov = torch.matmul(x_mean, x_mean.transpose(1, 2)) / (L - 1)
        try:
            inv_cov = torch.inverse(cov + torch.eye(C, device=x.device) * 1e-6)
        except torch.linalg.LinAlgError:  # 处理奇异矩阵
            return None
        diffs = x.unsqueeze(2) - x.unsqueeze(1)
        m_dist_sq = torch.einsum('bijd,bdk,bijk->bij', diffs, inv_cov, diffs)
        return -m_dist_sq

class SimilarityMask(nn.Module):
    """
    通过计算通道间的余弦相似度来生成注意力掩码.
    相似度越高，注意力越高.
    """
    def forward(self, x):
        # x: [B, C, L] (Batch, Channels, Length)
        B, C, L = x.shape
        if C <= 1:
            return None # 如果只有一个通道，则不需要mask

        # 1. 对每个通道的时间序列进行L2归一化
        #    这是计算余弦相似度的标准步骤
        # x_normed: [B, C, L]
        x_normed = F.normalize(x, p=2, dim=2)

        # 2. 使用批处理矩阵乘法计算所有通道对之间的余弦相似度
        #    (B, C, L) @ (B, L, C) -> (B, C, C)
        # similarity_matrix: [B, C, C]
        similarity_matrix = torch.bmm(x_normed, x_normed.transpose(1, 2))

        # 3. 返回相似度矩阵作为掩码
        #    这个矩阵可以直接被softmax使用，相似度越高的对(值接近1)会获得越高的注意力权重
        return similarity_matrix


class DynamicAttentionMask(nn.Module):
    """
    使用自注意力机制动态生成通道关系掩码.
    input:B,C,D  output:B,C,C
    """

    def __init__(self, d_model, n_heads=2):
        super().__init__()
        # 确保d_model可以被头数整除
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 学习Q, K的线性投影层
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, C, D]
        B, C, D = x.shape

        # 1. 生成Q, K
        # q, k: [B, C, H, d_k] -> [B, H, C, d_k] (H是头数)
        q = self.query(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算关系得分
        # scores: [B, H, C, C]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 3. 将得分作为掩码返回 (在实际使用中, 会直接softmax)
        # 为了与之前的模块兼容, 我们可以在外部应用softmax
        # 这里返回的是原始得分, 它反映了动态关系
        # 取平均头或者返回所有头
        return scores.mean(dim=1)  # -> [B, C, C]


class GraphConvMask(BaseMaskGenerator):
    """使用简单的图卷积网络来传播信息并生成掩码。"""

    def __init__(self, d_model):
        super().__init__(d_model)
        self.gcn_layer1 = nn.Linear(d_model, d_model)
        self.gcn_layer2 = nn.Linear(d_model, d_model)

    def forward(self, x, adj_matrix= None):
        # x: [B, C, D]
        if adj_matrix is None:
            bs, c, d = x.shape
            adj = torch.ones(bs, c, c, device=x.device)
        else:
            adj = adj_matrix.to(x.device)  # [C,C]

        # 简单的图卷积: A * X * W
        support1 = self.gcn_layer1(torch.bmm(adj, x))
        h1 = F.relu(support1)
        support2 = self.gcn_layer2(torch.bmm(adj, h1))

        # 将最终的节点表示用于计算关系
        mask = torch.bmm(support2, support2.transpose(1, 2))
        return mask

class LowRankDelta(nn.Module):
    """通过学习低秩因子来生成一个调整矩阵。"""
    def __init__(self, d_model, rank=16):
        # d_model: 特征维度, rank: 低秩矩阵的秩
        super().__init__()
        self.factor1 = nn.Linear(d_model, rank)
        self.factor2 = nn.Linear(d_model, rank)

    def forward(self, x):
        # x: [B, C, D]
        f1 = self.factor1(x)    # [B, C, rank]
        f2 = self.factor2(x)    # [B, C, rank]
        delta = torch.bmm(f1, f2.transpose(1, 2)) # [B, C, C]
        return delta

class AdditiveResidualMask(BaseMaskGenerator):
    """在全一矩阵基础上，学习一个加性残差项。"""

    def __init__(self, d_model, rank=16, alpha=0.1):
        # alpha: 控制残差项强度的超参数
        super().__init__(d_model)
        self.delta_generator = LowRankDelta(d_model, rank)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        # x: [B, C, D]
        B, C, D = x.shape

        # 1. 创建全一基底矩阵
        base_mask = torch.ones(B, C, C, device=x.device)

        # 2. 学习动态的残差项
        delta_mask = self.delta_generator(x)  # [B, C, C]
        delta_mask = torch.softmax(delta_mask, dim=-1)

        # 3. 返回最终掩码
        final_mask = base_mask + self.alpha * delta_mask
        return final_mask


class MaskedRNN(nn.Module):
    """一个封装了掩码输入逻辑的RNN模块."""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        # 核心是一个标准的GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [B, C, D] (C是通道数, D是d_model)
        # mask: [B, C, C]

        # 如果没有mask或只有一个通道, 直接通过RNN
        if mask is None:
            output, _ = self.rnn(x)
            return self.dropout(output)

        # 1. 将mask转换为注意力权重
        # weights: [B, C, C]
        mask_pd = pd.DataFrame(mask[0].detach().cpu().numpy())
        attn_weights = F.softmax(mask, dim=-1)

        # 2. 计算加权的输入
        # (B, C, C) @ (B, C, D) -> (B, C, D)
        # 每个通道的新特征是所有其他通道特征的加权和
        weighted_x = torch.bmm(attn_weights, x)

        # 3. 残差连接和归一化
        gated_x = self.norm(x + weighted_x)

        # 4. 将加权后的序列送入RNN
        # output: [B, C, D]
        output, _ = self.rnn(gated_x)

        return self.dropout(output)


class ParallelGatedRNN(nn.Module):
    """
    为每个通道并行创建一个专属的上下文视图，并用RNN处理。
    """

    def __init__(self, input_dim, hidden_dim, seq_len, num_layers=1, dropout=0.1):
        super().__init__()
        self.seg_len = 5
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.sensor_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sensor_squeeze = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            # nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cnn = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(seq_len)

    def forward(self, x, mask):
        x = x.permute(0,2,1)
        # x: [B, C, D], mask: [B, C, C]
        B, C, D = x.shape

        if mask is None:  # 如果没有mask，则退化为标准RNN
            output, _ = self.rnn(x)
            return self.dropout(output)

        # 1. 将相似度矩阵转换为每个通道的专属“门控权重”
        # gate_weights: [B, C, C]
        ones_mask = torch.ones_like(mask)
        # abs_mask = torch.abs(mask)
        # threshold_mask = (abs_mask >= 0.5).float()
        # gate_weights = mask
        gate_weights = F.softmax(mask, dim=-1)
        pd_weights = pd.DataFrame(gate_weights[0].detach().cpu().numpy())

        # 2. 创建并行的输入视图
        # 通过广播机制，为C个通道中的每一个都创建一个专属的输入
        # x (B, 1, C, D) * gate_weights (B, C, C, 1) -> gated_inputs (B, C, C, D)
        gated_inputs = x.unsqueeze(1) * gate_weights.unsqueeze(-1)

        # 3. 展平以进行批处理
        # gated_inputs_flat: [B*C, C, D]
        gated_inputs_flat = gated_inputs.reshape(B * C, C, D)
        gated_inputs_flat = gated_inputs_flat.permute(0, 2, 1) #[BC,D,C]

        # 4. 通过RNN处理所有并行的视图
        # rnn_outputs_flat: [B*C, C, D]
        gated_inputs_flat = gated_inputs_flat.reshape(-1, D//self.seg_len, self.seg_len, C).permute(0,2,1,3).reshape(-1, D//self.seg_len, C)
        rnn_input = self.sensor_embedding(gated_inputs_flat)
        rnn_outputs_flat = self.rnn(rnn_input)[0]
        rnn_outputs_flat = self.sensor_squeeze(rnn_outputs_flat).permute(0, 2, 1)
        rnn_outputs_flat = rnn_outputs_flat.reshape(-1, D, C).permute(0, 2, 1)

        # 5. 提取每个并行路径对应的目标输出
        # rnn_outputs: [B, C, C, D]
        rnn_outputs = rnn_outputs_flat.view(B, C, C, D)
        # 我们只关心第i个路径中第i个通道的输出
        # final_outputs = torch.sum(rnn_outputs, dim=1)
        # final_outputs = torch.max(rnn_outputs, dim=1).values
        # 使用einsum高效地提取对角线元素: (B, C, C, D) -> (B, C, D)
        # final_outputs = torch.einsum('bijd, ii -> bjd', rnn_outputs, torch.eye(C, device=x.device))
        final_outputs = torch.diagonal(rnn_outputs, offset=0, dim1=1, dim2=2).permute(0, 2, 1)
        final_outputs = self.channel(final_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        # final_outputs = self.norm(final_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        # final_outputs = self.cnn(final_outputs)

        # 6. 残差连接
        return final_outputs.permute(0, 2, 1)
