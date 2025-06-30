import torch.nn as nn
import torch
import torch.nn.functional as F

class SmoothedAnchorNorm(nn.Module):
    """
    通过平滑序列来获取鲁棒锚点，并进行归一化。
    在反向传播时，锚点的计算过程不参与梯度更新。
    input: b,s,c; output:b,s,c  b,1,c
    """

    def __init__(self, smoothing_kernel_size=25):
        super(SmoothedAnchorNorm, self).__init__()
        self.smoother = nn.AvgPool1d(
            kernel_size=smoothing_kernel_size,
            stride=1,
            padding=(smoothing_kernel_size - 1) // 2
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        x_permuted = x.permute(0, 2, 1)  # b,c,s

        with torch.no_grad():
            x_smoothed = self.smoother(x_permuted)  # b,c,s
        # anchor: [batch, features, 1]
        anchor = x_smoothed[:, :, -1].unsqueeze(-1)  # b,c,1

        # x_norm_permuted: [batch, features, seq_len]
        x_norm_permuted = x_permuted - anchor

        x_norm = x_norm_permuted.permute(0, 2, 1)  # b,s,c
        anchor_out = anchor.permute(0, 2, 1)  # b,1,c

        return x_norm, anchor_out

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

class ParallelRNN(nn.Module):
    def __init__(self, configs):
        super(ParallelRNN, self).__init__()
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
        self.cnn = nn.Conv1d(in_channels=self.d_model, out_channels=2 * self.d_model, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=self.d_model * 2, out_channels=self.d_model, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=self.enc_in, num_heads=1, dropout=self.dropout, batch_first=True)
        self.anchor_norm = SmoothedAnchorNorm()
        self.channel = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.d_model, self.enc_in),
            # nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.Mixer = ResBlock(self.sensors, self.seq_len, self.d_model, self.d_model, self.dropout)
        self.attention = nn.MultiheadAttention(embed_dim=self.enc_in, num_heads=1, dropout=self.dropout, batch_first=True)
        self.gamma = nn.Linear(self.enc_in, self.enc_in)
        self.beta = nn.Linear(self.enc_in, self.enc_in)

    def forecast(self, x_enc):
        batch_size = x_enc.shape[0]
        x_res = x_enc
        # x_enc = self.channel(x_enc)
        # seq_last = seq_last.permute(0, 2, 1)
        # seq_last = self.attention(seq_last, seq_last, seq_last)[0].permute(0, 2, 1)
        # x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x = self.rnn(x)[0]
        x = self.SensorSqueeze(x).reshape(batch_size, -1, self.sensors)
        x = self.norm(x)
        # x = x + seq_last
        # enc_output = self.SENet(x)
        return x

    def forward(self, x_enc):
        x_enc, seq_last = self.anchor_norm(x_enc)
        g = self.gamma(seq_last)
        b = self.beta(seq_last)
        enc_out = self.forecast(x_enc) * g + b
        enc_out = enc_out + seq_last
        #rul_head
        enc_out = self.projection(enc_out.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.sensors)
        enc_output = self.squeeze(enc_out_2d)
        return enc_output[:, -self.pred_len:]  # [B, L, D]