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

class ResBlock(nn.Module):
    def __init__(self, sensors, seq_len, t_model, c_model, dropout):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(t_model, seq_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(sensors, c_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(c_model, sensors),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)
        self.channel_conv  = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, padding=0)

        self.norm = nn.BatchNorm1d(seq_len)
    def forward(self, x):
        # x: [B, L, D]
        x_tprl = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x_chnl = self.channel(x)
        # x_aton = self.attention_layer(x)


        x_out = x + self.temporal_conv(x_tprl.unsqueeze(1)).squeeze(1) + self.channel_conv(x_chnl.unsqueeze(1)).squeeze(1)
        # x_out = x + x_tprl+ x_chnl
        x_out = self.norm(x_out)
        return x_out

class Simple_SegRNN(nn.Module):
    def __init__(self, configs):
        super(Simple_SegRNN, self).__init__()
        self.name = 'LSTM_pTSMixer_GA'
        self.layer = configs.e_layers
        self.accept_window = configs.seq_len
        self.sensors = configs.enc_in
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.seq_len = configs.seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.dropout = configs.dropout
        self.lstm_layer_num = configs.d_layers

        # self.lstm = nn.LSTM(input_size=self.sensors, hidden_size=self.sensors, num_layers=self.lstm_layer_num, batch_first=True)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num, batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.SensorsEmbedding = nn.Sequential(
            nn.Linear(self.sensors, self.d_model),
            nn.ReLU()
        )
        self.SensorSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )

        self.model = nn.ModuleList(
            [ResBlock(self.sensors, self.seq_len, self.d_model, self.d_model, self.dropout)
             for _ in range(configs.d_layers)]
        )
        self.norm = nn.BatchNorm1d(self.seq_len)

        # self.attention_layer = New_AttentionBlockBranch(self.enc_in, self.seq_len)

        self.pred_len = 1
        # self.projection = nn.Linear(seq_len, pred_len)
        # self.squeeze = nn.Linear(sensors, pred_len)
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
        self.SENet = APRULHead(self.seq_len, self.sensors, self.d_model, self.dropout)
        self.mixed = nn.Sequential(
            nn.Linear(self.enc_in * self.seg_num_x, self.enc_in * self.seg_num_x * 2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.enc_in * self.seg_num_x * 2, self.enc_in * self.seg_num_x),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.conv_layer = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(3, 3), padding=1)

    def forecast(self, x_enc):
        batch_size = x_enc.shape[0]
        # x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3) #b,s,n,d
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x = self.rnn(x)[0]
        x = self.SensorSqueeze(x).reshape(batch_size, -1, self.sensors)
        x = self.norm(x)
        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.sensors)
        enc_output = self.squeeze(enc_out_2d)



        # x = self.valueEmbedding(x_enc.reshape(-1, self.seg_num_x, self.seg_len)) #bc,n,d
        #mlp-mixer
        # x = x.reshape(batch_size, self.enc_in * self.seg_num_x, self.d_model)
        # x = self.mixed(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = x.reshape(-1, self.seg_num_x, self.d_model)
        #2dCNN
        # x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.d_model).permute(0, 3, 1, 2) #b,d,c,n
        # x = self.conv_layer(x).permute(0, 2, 3, 1).reshape(-1, self.seg_num_x, self.d_model)
        # x = self.rnn(x)[0]#bc, n, d
        # x = self.predict(x).reshape(batch_size, self.sensors, -1).permute(0, 2, 1) #b,s,c
        # x = self.lstm(x_enc)[0] + x_enc
        # x_sliced = x[:, -self.accept_window:, :].contiguous()
        # x: [B, L, D]
        # for i in range(self.layer):
        #     x_sliced = self.model[i](x_sliced)
        # x_sliced = self.norm(x_sliced)
        # x_sliced = self.attention_layer(x_sliced)
        # enc_output = self.SENet(x_sliced)
        # enc_out = self.projection(x_sliced.transpose(1, 2)).transpose(1, 2)
        # enc_out_2d = enc_out.view(-1, self.sensors)
        # enc_output = self.squeeze(enc_out_2d)
        return enc_output

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:]  # [B, L, D]