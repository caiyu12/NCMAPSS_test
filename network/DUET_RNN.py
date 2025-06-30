import torch
import torch.nn as nn
from layers import LinearExtractor, SimilarityMask, MaskedRNN


class FastTrendGate(nn.Module):
    """高效的趋势分析门控，用于判断数据类型."""

    def __init__(self, cv_threshold=1.5):
        super().__init__()
        self.cv_threshold = cv_threshold

    @torch.no_grad()
    def forward(self, x_batch):
        # x_batch: [B, L, C]
        if x_batch.shape[1] < 2: return 0
        diff = torch.diff(x_batch, dim=1)
        trend_scores = torch.sum(torch.abs(diff), dim=1)
        mean_scores = torch.mean(trend_scores, dim=1)
        std_devs = torch.std(trend_scores, dim=1)
        safe_mean_scores = mean_scores + 1e-9
        cv = std_devs / safe_mean_scores
        mean_batch_cv = torch.mean(cv)
        return 1 if mean_batch_cv > self.cv_threshold else 0


class AdaptiveDuetRNN(nn.Module):
    """
    使用MaskedRNN来建模通道关系的自适应DUET模型.
    """

    def __init__(self, configs, cv_threshold=1.5):
        super().__init__()

        # 1. 自适应门控和归一化模块
        self.gate = FastTrendGate(cv_threshold)
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.rnn_layers = 2

        # 2. DUET核心模块
        # 阶段一：时序特征提取
        self.linear_extractor = LinearExtractor(self.seq_len, self.d_model)

        self.similarity_mask = SimilarityMask()
        self.channel_rnn = MaskedRNN(
            input_dim=self.d_model,
            hidden_dim=self.d_model,
            num_layers=self.rnn_layers,
            dropout=self.dropout
        )

        # 3. RUL预测头
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1)
        )

    def forward(self, x):
        # x: [B, L, C]

        # --- 自适应归一化 ---
        norm_type = self.gate(x)
        seq_last = x[:, -1:, :].detach()
        normalized_x = x - seq_last
        # if norm_type == 1:  # 处理后数据 -> 减去最后一个值
        #     seq_last = x[:, -1:, :].detach()
        #     normalized_x = x - seq_last
        # else:  # 原始数据 -> 不归一化
        #     seq_last = 0
        #     normalized_x = x

        # --- DUET 阶段一：时序特征提取 ---
        # temporal_feature: [B, d_model, C]
        temporal_feature = self.linear_extractor(normalized_x)

        # --- DUET 阶段二：通道关系建模 ---
        # -> [B, C, d_model], 将C个通道作为序列长度
        temporal_feature_swapped = temporal_feature.transpose(1, 2)

        # -> [B, C, C], 计算通道相似性掩码
        attn_mask = self.similarity_mask(normalized_x.transpose(1, 2))

        # channel_feature: [B, C, d_model]
        channel_feature = self.channel_rnn(temporal_feature_swapped, mask=attn_mask)

        # --- RUL预测头 ---
        # 使用最后一个隐状态或平均池化来聚合信息
        aggregated_feature = torch.mean(channel_feature, dim=1)  # -> [B, d_model]

        # rul_prediction: [B, 1]
        rul_prediction = self.head(aggregated_feature)

        return rul_prediction

