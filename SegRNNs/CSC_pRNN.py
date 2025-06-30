import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPoolingRULHead(nn.Module):
    def __init__(self, seq_len, enc_in, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.attention_layer = nn.Linear(self.enc_in, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.rul_projection = nn.Linear(int(self.enc_in), 1)

    def forward(self, enc_out):
        attention_scores = self.attention_layer(enc_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(enc_out * attention_weights, dim=1)
        context_vector = self.dropout(context_vector)
        rul_output = self.rul_projection(context_vector)
        return rul_output

class CNN_layer(nn.Module):
    def __init__(self, enc_in, num_filters, kernel_size):
        super(CNN_layer, self).__init__()
        self.enc_in = enc_in
        self.conv1 = nn.Conv1d(in_channels=enc_in, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        # x:b,enc_in,s
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x) # b,nf*2,s//4
        return x


class CSC_pRNN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """
    def __init__(self, configs):
        super(CSC_pRNN, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers
        self.num_filters = configs.CNN_size
        self.time_features = 2
        self.fault_features = 4

        self.task_name = configs.task_name
        if self.task_name in ['classification', 'anomaly_detection', 'imputation', 'rul_prediction']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        # self.seg_num_y = self.pred_len // self.seg_len

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.time_features, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.fault_features, self.d_model // 2))

        # 添加交叉注意力层
        self.s_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1)
        self.c_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=1)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.sensor_cnn = CNN_layer(self.enc_in - 4, self.num_filters, 3)
        self.cov_cnn = CNN_layer(4, self.num_filters, 3)
        self.feature_squeeze = nn.Sequential(
            nn.Linear(self.num_filters * 2, self.fault_features),
            # nn.ReLU()
        )
        self.nor_MLP = nn.Linear(self.enc_in - 4, self.fault_features)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.fault_features * configs.seq_len, configs.num_class)

        if self.task_name == 'rul_prediction':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.rul_projection = nn.Linear(self.fault_features * self.seg_len * self.time_features, 1)
            self.rul_head = AttentionPoolingRULHead(self.seq_len, self.enc_in, configs.dropout)

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
        sensors = x[:, :, 4:]     # [b, s, c-4]

        # 对传感器变量进行段分割和嵌入
        sensors = sensors.permute(0, 2, 1)  # [b, c-4, s]
        sensors = self.sensor_cnn(sensors) # b, nf*2, s//4
        sensors = self.feature_squeeze(sensors.permute(0, 2, 1)).permute(0, 2, 1) #b,t,s//4
        sensors = sensors.reshape(-1, self.seg_num_x//4, self.seg_len)  # [b*t, seg_num_x, seg_len]
        sensors_emb = self.valueEmbedding(sensors)  # [b*t, seg_num_x, d_model]

        # 对协变量进行段分割和嵌入
        covariates = covariates.permute(0, 2, 1)  # [b, 4, s]
        covariates_pre = covariates.reshape(-1, self.seg_num_x, self.seg_len)  # [b*4, seg_num_x, seg_len]
        covariates_emb = self.valueEmbedding(covariates_pre)  # [b*4, seg_num_x, d_model]

        # 对传感器嵌入进行 RNN 编码
        _, sensors_hn = self.rnn(sensors_emb)  # [1, b*t, d_model]

        # 对协变量嵌入进行 RNN 编码
        _, covariates_hn = self.rnn(covariates_emb)  # [1, b*4, d_model]

        # 调整形状以适应交叉注意力
        sensors_hn = sensors_hn.view(batch_size, self.fault_features, self.d_model)  # [b, t, d_model]
        covariates_hn = covariates_hn.view(batch_size, 4, self.d_model)    # [b, 4, d_model]

        # 转置以适应 nn.MultiheadAttention
        sensors_hn = sensors_hn.permute(1, 0, 2)  # [t, b, d_model]
        covariates_hn = covariates_hn.permute(1, 0, 2)  # [4, b, d_model]
        sensors_attn, _ = self.s_attention(sensors_hn, sensors_hn, sensors_hn)
        sensors_hn = sensors_attn + sensors_hn
        sensors_hn = self.layer_norm(sensors_hn)
        # 交叉注意力：传感器作为查询，协变量作为键和值
        attn_output, _ = self.c_attention(sensors_hn, covariates_hn, covariates_hn)  # [t, b, d_model]

        # 转回 [b, c-4, d_model]
        attn_output = attn_output.permute(1, 0, 2)

        # 结合注意力输出和传感器 hn
        combined_hn = sensors_hn.permute(1, 0, 2) + attn_output  # [b, t, d_model]

        # 后续处理：生成预测序列
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.fault_features, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.time_features, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)  # [b*(c-4), m, d]

        _, hy = self.rnn(pos_emb, combined_hn.repeat(1, 1, self.time_features).view(1, -1, self.d_model))  # [1, b*(c-4)*m, d]
        for i in range(self.num_layers - 1):
            _, hy = self.rnn(pos_emb, hy)

        # 生成输出序列
        y = self.predict(hy).view(batch_size, self.fault_features, self.time_features*self.seg_len)  # [b, c-4, s]
        y = y.permute(0, 2, 1)
        # 补齐协变量部分（这里简单重复最后一个协变量值，可根据需求调整）
        # covariates = covariates.permute(0, 2, 1)
        # covariates_out = covariates[:, -1:, :].repeat(1, self.pred_len, 1)  # [b, s, 4]
        # y = torch.cat([covariates_out, y.permute(0, 2, 1)], dim=2)  # [b, s, c]

        # denorm
        y = y + self.nor_MLP(seq_last[:, :, 4:])
        return y

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def rul_prediction(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.rul_projection(output)
        # output = self.rul_head(enc_out)  # 如果需要使用注意力头，可取消注释
        return output

    def forward(self, x_enc):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out
        if self.task_name == 'rul_prediction':
            dec_out = self.rul_prediction(x_enc)
            return dec_out
        return None