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

class CA_pRNN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(CA_pRNN, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len

        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation' or self.task_name == 'rul_prediction':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in - 4, self.d_model // 2))
        self.s_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4,batch_first=True)
        self.c_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4,batch_first=True)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
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
                (configs.enc_in-4) * configs.seq_len, 1)
            # self.rul_head = AttentionPoolingRULHead(self.seq_len, self.enc_in, configs.dropout)

    def encoder(self, x):
        # x: [b, s, c]
        batch_size = x.size(0)
        seq_len = x.size(1)
        enc_in = x.size(2)

        # normalization
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        # 分离协变量和传感器变量
        covariates = x[:, :, :4]  # [b, s, 4]
        sensors = x[:, :, 4:]  # [b, s, c-4]

        sensors = sensors.permute(0, 2, 1)  # [b, c-4, s]
        sensors = self.valueEmbedding(sensors.reshape(-1, self.seg_num_x, self.seg_len)) #[b*(c-4), n, d]
        _, sensors_hn = self.rnn(sensors)  # [b*(c-4),1,d]

        covariates = covariates.permute(0, 2, 1)  # [b, 4, s]
        covariates = self.valueEmbedding(covariates.reshape(-1, self.seg_num_x, self.seg_len)) #[b*4, n, d]
        _, covariates_hn = self.rnn(covariates) # [b*4, 1, d]

        sensors_hn = sensors_hn.view(batch_size, enc_in - 4, self.d_model)  # [b, c-4, d]
        covariates_hn = covariates_hn.view(batch_size, 4, self.d_model)  # [b, 4, d]

        sensors_attn, _ = self.s_attention(sensors_hn, sensors_hn, sensors_hn) # [b, c-4, d]
        sensors_hn = sensors_attn + sensors_hn
        sensors_hn = self.layer_norm(sensors_hn)

        attn_output, _ = self.c_attention(sensors_hn, covariates_hn, covariates_hn) # [b, c-4, d]
        combined_hn = sensors_hn + attn_output # [b, c-4, d]

        # 1，d//2 -> 1,1，d//2 -> c - 4, 1, d//2
        # c-4,d//2 -> c,1,d//2
        # c-4,1,d -> c-4,1,d -> b*(c-4),1,d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in - 4, 1, 1),
            self.channel_emb.unsqueeze(1)
        ], dim=-1)
        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, combined_hn.view(1, -1, self.d_model))  # bc',1,d  1,bc',d
        for i in range(self.num_layers - 1):
            _, hy = self.rnn(pos_emb, hy)

        # 1,bc,d -> b,c,d -> b,c,p
        y = self.predict(hy).view(-1, self.enc_in - 4, self.pred_len)
        y = y.permute(0, 2, 1) + seq_last[:, :, 4:]
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
        enc_out = self.encoder(x_enc) + x_enc[:, :, 4:]
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

