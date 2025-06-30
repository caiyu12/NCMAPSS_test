import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(max_len, embed_dim)) #c,s

    def forward(self, x, dim=-1):
        d_model = x.size(dim)
        batch_size = x.size(0)
        pos_embed = self.pos_embed[:d_model, :].unsqueeze(0).repeat(batch_size,1,1).permute(0,2,1)
        return x + pos_embed

class TimeSelfAttention(nn.Module):
    def __init__(self, seq_len, channels, n_heads, dropout_rate=0.1):
        super().__init__()
        self.pos_embed = LearnedPositionalEmbedding(seq_len, channels)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=dropout_rate,batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x:b,c,s
        x = self.pos_embed(x, dim=2)
        x = x.permute(0,2,1) # b,s,c
        attn_output, _ = self.attention(x, x, x) # b,s,c
        x = x + self.dropout(attn_output)
        x = self.norm(x).permute(0, 2, 1)
        return x

class ChannelSelfAttention(nn.Module):
    def __init__(self, seq_len, channels, n_heads, dropout_rate=0.1):
        super().__init__()
        self.pos_embed = LearnedPositionalEmbedding(channels, seq_len)
        self.attention = nn.MultiheadAttention(embed_dim=seq_len, num_heads=n_heads, dropout=dropout_rate,batch_first=True)
        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pos_embed(x, dim=2)
        x_attn = x.permute(0, 2, 1)
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + self.dropout(attn_output).permute(0, 2, 1)
        x = self.norm(x.permute(0,2,1))
        return x

class AttentionBlock(nn.Module):
    def __init__(self, seq_len, channels, n_heads, dropout_rate=0.1):
        super().__init__()
        self.time_attention = TimeSelfAttention(seq_len, channels, 1, dropout_rate)
        self.channel_attention = ChannelSelfAttention(seq_len, channels, n_heads, dropout_rate)

    def forward(self, x):
        x = self.time_attention(x)
        x = self.channel_attention(x)
        return x

class AttentionPoolingRULHead(nn.Module):
    def __init__(self, seq_len, enc_in, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        # 用于计算注意力分数的查询向量（或使用更复杂的Q, K, V）
        # 简单起见，这里使用一个线性层来学习每个时间步的重要性
        self.attention_layer = nn.Linear(self.enc_in, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.rul_projection = nn.Linear(int(self.enc_in), 1) # 输出1维的RUL值

    def forward(self, enc_out):
        # enc_out shape: (batch_size, seq_len, d_model)

        # 计算每个时间步的注意力分数
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, 1)
        attention_scores = self.attention_layer(enc_out)
        # 应用softmax得到权重
        # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # 使用注意力权重对序列进行加权求和
        # (batch_size, seq_len, d_model) * (batch_size, seq_len, 1) -> (batch_size, seq_len, d_model)
        # sum over seq_len dimension -> (batch_size, d_model)
        context_vector = torch.sum(enc_out * attention_weights, dim=1)

        # 应用 dropout
        context_vector = self.dropout(context_vector)

        # 通过最终线性层进行 RUL 预测
        # (batch_size, d_model) -> (batch_size, 1)
        rul_output = self.rul_projection(context_vector)
        return rul_output

class TC_SA_SegRNN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(TC_SA_SegRNN, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers

        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation' or self.task_name == 'rul_prediction':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.SelfAtten = AttentionBlock(self.seq_len, self.enc_in, configs.n_heads, configs.dropout)


        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

        if self.task_name == 'rul_prediction':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.rul_projection = nn.Linear(
                configs.enc_in * configs.seq_len, 1)
            self.rul_head = AttentionPoolingRULHead(self.seq_len, self.enc_in, configs.dropout)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) # b,c,s

        x = self.SelfAtten(x)
        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x) # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) # bcm,1,d  1,bcm,d
        for i in range(self.num_layers - 1):
            _,hy = self.rnn(pos_emb, hy)

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def rul_prediction(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.rul_projection(output)
        # output = self.rul_head(enc_out)
        return output

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'rul_prediction':
            dec_out = self.rul_prediction(x_enc)
            return dec_out
        return None

