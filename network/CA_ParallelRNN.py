import torch.nn as nn
import torch
import torch.nn.functional as F

class APRULHead(nn.Module):
    def __init__(self, seq_len, enc_in, d_model=64, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.feature_projection = nn.Linear(enc_in, d_model)
        self.temporal_attention = nn.Linear(d_model, 1)
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),  # 压缩到较小维度
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),  # 扩展回 d_model
            nn.Sigmoid()  # 生成通道权重
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.rul_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # 输入 x: [batch_size, seq_len, enc_in]
        x = self.feature_projection(x)  # [batch_size, seq_len, d_model]
        temporal_scores = self.temporal_attention(x)  # [batch_size, seq_len, 1]
        temporal_weights = F.softmax(temporal_scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(x * temporal_weights, dim=1)  # [batch_size, d_model]
        channel_weights = self.channel_attention(context_vector)  # [batch_size, d_model]
        context_vector = context_vector * channel_weights  # [batch_size, d_model]
        context_vector = self.dropout(context_vector)

        rul_output = self.rul_projection(context_vector)  # [batch_size, 1]
        return rul_output

class CA_ParallelRNN(nn.Module):
    def __init__(self, configs):
        super(CA_ParallelRNN, self).__init__()
        self.name = 'LSTM_pTSMixer_GA'
        self.layer = configs.e_layers
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.seq_len = configs.seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.dropout = configs.dropout
        self.lstm_layer_num = configs.d_layers
        self.cov_div  = configs.cov_div
        self.task = configs.loadmode
        if self.task == 'normal':
            self.sensors = self.enc_in - self.cov_div
        elif self.task == 'cheat1':
            self.sensors = self.cov_div

        # self.lstm = nn.LSTM(input_size=self.sensors, hidden_size=self.sensors, num_layers=self.lstm_layer_num, batch_first=True)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num, batch_first=True)
        self.SensorsEmbedding = nn.Sequential(
            nn.Linear(self.sensors, self.d_model),
            nn.ReLU()
        )
        self.SensorSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.sensors),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )
        self.norm = nn.BatchNorm1d(self.seq_len)
        self.pred_len = 1
        self.projection = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze    = nn.Sequential(
            nn.Linear(self.sensors, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.SENet = APRULHead(self.seq_len, self.enc_in, self.d_model, self.dropout)
        self.s_attention = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=1, batch_first=True)
        self.c_attention = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.seq_len)

    def forecast(self, x_enc): #b,seq,c
        batch_size = x_enc.shape[0]
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x = self.rnn(x)[0] # bs,n,d
        x = self.SensorSqueeze(x).reshape(batch_size, -1, self.enc_in) #b,seq,c
        x = self.norm(x)
        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.enc_in)
        enc_output = self.squeeze(enc_out_2d)
        return enc_output

    def rul_prediction(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        enc_in = x.size(2)
        new_enc = enc_in
        if self.task == 'normal':
            covariates = x[:, :, :self.cov_div]  # [b, s, v]
            sensors = x[:, :, self.cov_div:]  # [b, s, c-v]
            new_enc = enc_in - self.cov_div
        elif self.task == 'cheat1':
            covariates = x[:, :, self.cov_div:]  # [b, s, c-v]
            sensors = x[:, :, :self.cov_div]  # [b, s, v]
            new_enc = self.cov_div
        sensors = sensors.permute(0, 2, 1)
        covariates = covariates.permute(0, 2, 1)
        sen_emb, _ = self.s_attention(sensors, sensors, sensors)
        sen_emb = self.layer_norm(sen_emb+sensors)
        sen_cross, _ = self.c_attention(sen_emb, covariates, covariates)
        sen_cross = sen_cross + sen_emb
        sen_cross = sen_cross.permute(0, 2, 1)
        x_enc = sen_cross.reshape(batch_size, self.seg_num_x, self.seg_len, new_enc).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)
        x = self.rnn(x)[0]  # bs,n,d
        x = self.SensorSqueeze(x).reshape(batch_size, -1, self.sensors)  # b,seq,c-v
        x = self.norm(x)
        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.sensors)
        enc_output = self.squeeze(enc_out_2d)
        return enc_output



    def forward(self, x_enc):
        # dec_out = self.forecast(x_enc)
        dec_out = self.rul_prediction(x_enc)
        return dec_out[:, -self.pred_len:]  # [B, L, D]