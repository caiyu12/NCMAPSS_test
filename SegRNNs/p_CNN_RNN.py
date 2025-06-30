import torch
import torch.nn as nn
import torch.nn.functional as F

class p_CNN_RNN(nn.Module):
    def __init__(self, configs):
        super(p_CNN_RNN, self).__init__()

        # 配置参数
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers
        self.task_name = configs.task_name
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.num_heads = configs.n_heads

        # Value Embedding
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        # GRU 层
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)

        # CNN 层（对通道维度 enc_in 操作）
        self.cnn = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in * 2, kernel_size=3, padding=1)

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.num_heads)

        # Position Embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        # 预测层
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seq_len)
        )

        # RUL 预测层
        if self.task_name == 'rul_prediction':
            self.rul_projection = nn.Linear(self.enc_in * self.seq_len, 1)

    def encoder(self, x):
        batch_size = x.size(0)

        # 输入归一化
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        # Value Embedding
        x = x.reshape(batch_size * self.enc_in, self.seg_num_x, self.seg_len)
        x = self.valueEmbedding(x)  # (batch_size * enc_in, seg_num_x, d_model)

        # GRU 层
        _, hn = self.rnn(x)  # hn: (1, batch_size * enc_in, d_model)

        # 处理隐藏状态
        hn = hn.view(batch_size, self.enc_in, self.d_model)  # (batch_size, enc_in, d_model)

        # CNN 处理（在 enc_in 维度上）
        hn_cnn = self.cnn(hn)  # (batch_size, enc_in*2, d_model)

        # 自注意力处理
        hn_attn, _ = self.self_attn(hn_cnn.permute(1, 0, 2), hn_cnn.permute(1, 0, 2), hn_cnn.permute(1, 0, 2))  # (enc_in*2, batch_size, d_model)
        hn_attn = hn_attn.permute(1, 2, 0)  # (batch_size, d_model, 2*enc_in)
        hn_attn = hn_attn[:, :, self.enc_in:].contiguous().permute(0, 2, 1) #b,c,d

        # 转回原始形状
        new_hn = hn_attn.reshape(1, batch_size * self.enc_in, self.d_model)  # (1, batch_size * enc_in, d_model)

        # Position Embedding
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # (enc_in, 1, d_model // 2)
            self.channel_emb.unsqueeze(1)  # (enc_in, 1, d_model // 2)
        ], dim=-1)  # (enc_in, 1, d_model)
        pos_emb = pos_emb.view(-1, 1, self.d_model).repeat(batch_size, 1, 1)  # (batch_size * enc_in, 1, d_model)

        # 后续 GRU 层
        _, hy = self.rnn(pos_emb, new_hn)
        for i in range(self.num_layers - 1):
            _, hy = self.rnn(pos_emb, hy)

        # 预测
        y = self.predict(hy).view(batch_size, self.enc_in, self.seq_len)
        y = y.permute(0, 2, 1) + seq_last
        return y

    def rul_prediction(self, x_enc):
        enc_out = self.encoder(x_enc) + x_enc  # (batch_size, seq_len, enc_in)
        output = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, seq_len * enc_in)
        output = self.rul_projection(output)  # (batch_size, 1)
        return output

    def forward(self, x_enc):
        if self.task_name == 'rul_prediction':
            return self.rul_prediction(x_enc)
        return None