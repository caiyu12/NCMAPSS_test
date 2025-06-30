import torch
import torch.nn as nn
import torch.nn.functional as F

class TSMixer(nn.Module):
    def __init__(self, n, d_model):
        super(TSMixer, self).__init__()
        self.time_mixing = nn.Sequential(
            nn.Linear(n, n),
            nn.ReLU()
        )
        self.feature_mixing = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, n, d_model)
        # Time mixing
        x_time = x.permute(0, 2, 1)  # (b, d_model, n)
        x_time = self.time_mixing(x_time)  # (b, d_model, n)
        x_time = x_time.permute(0, 2, 1)  # (b, n, d_model)
        x = self.norm1(x + x_time)

        # Feature mixing
        x_feat = self.feature_mixing(x)  # (b, n, d_model)
        x = self.norm2(x + x_feat)

        return x

class poolRNN3(nn.Module):
    def __init__(self, configs):
        super(poolRNN3, self).__init__()

        # 从 configs 中提取参数
        self.seq_len = configs.seq_len  # 输入序列长度 (window_size)
        self.enc_in = configs.enc_in    # 输入特征维度 (c)
        self.d_model = configs.d_model  # 模型维度
        self.dropout = configs.dropout
        self.task_name = configs.task_name

        # 其他必要参数
        self.seg_len = configs.seg_len  # 池化后的目标序列长度
        self.num_heads = configs.n_heads  # 自注意力头数

        # 池化层，将 window_size 降到 n
        self.pool = nn.AvgPool1d(kernel_size=self.seg_len, stride=self.seg_len)

        # 输入特征投影到 d_model
        self.input_proj = nn.Conv1d(in_channels=self.enc_in, out_channels=self.d_model, kernel_size=3, padding=1)

        # 第一个 GRU 层
        self.gru = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)

        # 处理 hn 的 CNN 层
        self.cnn = nn.Conv1d(in_channels=self.d_model, out_channels=2 * self.d_model, kernel_size=1)

        # 处理 hn 的自注意力层
        self.self_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1)

        # TSMixer 模块处理 output
        self.tsmixer = TSMixer(n=self.seq_len // self.seg_len, d_model=self.d_model)

        # RUL 预测的线性层
        self.rul_projection = nn.Linear(self.d_model, 1)

    def encoder(self, x_enc):
        batch_size = x_enc.size(0)
        # 输入形状: (b, window_size, c)

        # 第1步：池化
        x = x_enc.permute(0, 2, 1)  # (b, c, window_size)
        x_pooled = self.pool(x)     # (b, c, n)
        x_pooled = x_pooled.permute(0, 2, 1)  # (b, n, c)

        # 第2步：特征投影
        x_pooled = self.input_proj(x_pooled.permute(0, 2, 1)).permute(0, 2, 1)  # (b, n, d_model)

        # 第3步：第一个 GRU
        output, hn = self.gru(x_pooled)  # output: (b, n, d_model), hn: (1, b, d_model)

        # 第4步：处理 hn（保持原有方式）
        hn_cnn = self.cnn(hn.permute(1, 2, 0))  # (b, 2*d_model, 1)
        hn_cnn = hn_cnn.permute(1, 0, 2)       # (2*d_model, b, 1)
        attn_output, _ = self.self_attn(hn_cnn, hn_cnn, hn_cnn)  # (2*d_model, b, 1)
        attn_output = attn_output.permute(2, 1, 0)  # (1, b, 2*d_model)
        new_hn = attn_output[:, :, self.d_model:].contiguous()  # (1, b, d_model)

        # 第5步：使用 TSMixer 处理 output
        new_output = self.tsmixer(output)  # (b, n, d_model)

        # 第6步：第二个 GRU
        output2, hn2 = self.gru(new_output, new_hn)  # output2: (b, n, d_model), hn2: (1, b, d_model)

        # 返回最终隐藏状态
        return hn2.squeeze(0)  # (b, d_model)

    def rul_prediction(self, x_enc):
        # 编码输入
        enc_out = self.encoder(x_enc)  # (b, d_model)

        # 预测 RUL
        output = self.rul_projection(enc_out)  # (b, 1)
        return output

    def forward(self, x_enc):
        if self.task_name == 'rul_prediction':
            return self.rul_prediction(x_enc)
        # 其他任务的占位符
        return None