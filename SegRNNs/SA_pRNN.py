import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入 x: (batch_size, seq_len, d_model)
        # 调整为 (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)
        # 恢复形状 (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)
        return x

class SA_pRNN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(SA_pRNN, self).__init__()

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

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )
        self.seq_attention = SelfAttentionBlock(self.enc_in, n_heads=1, dropout_rate=self.dropout)
        self.positional_encoding = PositionalEncoding(self.enc_in, max_len=self.seq_len)

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
            # self.rul_head = AttentionPoolingRULHead(self.seq_len, self.enc_in, configs.dropout)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) # b,c,s

        #embedding
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len)) #b,c,s -> bc,n,d
        # x_attn = self.positional_encoding(x)
        # x_attn = self.seq_attention(x_attn) # bc,n,d

        # encoding
        _, hn = self.rnn(x) # bc,1,d  1,bc,d

        # hn = hn.reshape(batch_size, -1, self.d_model)
        # _, hn = self.rnn(hn)
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # 1，d//2 -> 1,1，d//2 -> c, 1, d//2
            self.channel_emb.unsqueeze(1)  # c,d//2 -> c,1,d//2
        ], dim=-1)  # c,1,d
        pos_emb = pos_emb.view(-1, 1, self.d_model).repeat(batch_size,1,1) # c,1,d -> bc,1,d

        _, hy = self.rnn(pos_emb, hn.view(1, -1, self.d_model)) # bc,1,d  1,bc,d
        for i in range(self.num_layers - 1):
            _,hy = self.rnn(pos_emb, hy)


        # 1,bc,d -> b,c,d -> b,c,p
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
        enc_out = self.encoder(x_enc) + x_enc
        enc_out = self.positional_encoding(enc_out)
        enc_out = self.seq_attention(enc_out)
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

