import torch
import torch.nn as nn
import torch.nn.functional as F

class New_AttentionBlockBranch(nn.Module):
    def __init__(self, sensors, seq_len):
        super().__init__()
        '''
        /*--------layer-1---------------------------------------------------------------------*/
        '''
        self.sensors = sensors
        self.layer1_conv2d_1by1 = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=1, stride=1, padding=0,
            bias=False
        )
        '''
        /*--------layer-2---------------------------------------------------------------------*/
        '''
        self.layer2_conv2d_1by1_Res = nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=1, stride=1, padding=0,
            bias=True
        )
        # self.layer2_pool_Sensors = nn.MaxPool1d(kernel_size=seq_len, stride=1) # Windows Size
        # self.layer2_pool_TimeWin = nn.AvgPool1d(kernel_size=14, stride=1) # Effective Sensors
        __factor = 2
        self.factor_Sensors = int(sensors/__factor)
        self.factor_TimeWin = int(seq_len/__factor)
        self.layer2_pool_Sensors = nn.AdaptiveAvgPool1d(output_size=self.factor_Sensors)
        self.layer2_pool_TimeWin = nn.AdaptiveAvgPool1d(output_size=self.factor_TimeWin)
        __feature_size = self.sensors*self.factor_Sensors + seq_len*self.factor_TimeWin

        self.layer2_linear_seq = nn.Sequential(
            nn.Linear(in_features=__feature_size   , out_features=int(__feature_size/2)), # 22 = (30 + 14)/2
            nn.Dropout(p=0.1),

            nn.Linear(in_features=int(__feature_size/2), out_features=__feature_size),
            nn.Dropout(p=0.1),
            nn.ReLU(),
        )
        self.layer2_linear_Sensors = nn.Linear(in_features=sensors*self.factor_Sensors, out_features=sensors)
        self.layer2_linear_TimeWin = nn.Linear(in_features=seq_len*self.factor_TimeWin, out_features=seq_len)
        '''
        /*--------activation---------------------------------------------------------------------*/
        '''
        self.Sigmoid = nn.Sigmoid()
        self.TanH = nn.Tanh()
        self.ReLU = nn.ReLU()
        '''
        /*-----------------------------------------------------------------------------*/
        '''
        self.BatchNorm = nn.BatchNorm1d(seq_len)

    def forward(self, x : torch.tensor) -> torch.tensor:
        x_res    = self.layer1_conv2d_1by1(x.unsqueeze(dim=1))
        x_layer1 = x.unsqueeze(dim=1)
        x_layer1_3d = torch.flatten(x_layer1, start_dim=1, end_dim=2)

        x_layer2_SensorsAttention = self.layer2_pool_Sensors(x_layer1_3d.permute(0, 2, 1)) #(N, Sensors, 2)
        x_layer2_TimeWinAttention = self.layer2_pool_TimeWin(x_layer1_3d)                  #(N, TimeWin, int(seq_len/7))
        shape_Sensors = x_layer2_SensorsAttention.shape
        shape_TimeWin = x_layer2_TimeWinAttention.shape

        x_layer2_SensorsAttention = torch.flatten(x_layer2_SensorsAttention, start_dim=1) #(N, Sensors)
        x_layer2_TimeWinAttention = torch.flatten(x_layer2_TimeWinAttention, start_dim=1) #(N, TimeWin)
        x_layer2_linear = self.layer2_linear_seq(torch.cat([x_layer2_SensorsAttention, x_layer2_TimeWinAttention], dim=1))
        #(N, Sensors+TimeWin)->(N, Sensors+TimeWin)
        x_layer2_SensorsAttention = self.layer2_linear_Sensors(x_layer2_linear[:, :self.sensors*self.factor_Sensors]).unsqueeze(dim=2)
        x_layer2_TimeWinAttention = self.layer2_linear_TimeWin(x_layer2_linear[:, self.sensors*self.factor_Sensors:]).unsqueeze(dim=2)

        x_layer2_SensorsAttention_result = self.Sigmoid(x_layer2_SensorsAttention)
        x_layer2_TimeWinAttention_result = self.Sigmoid(x_layer2_TimeWinAttention)
        # x_layer2_SensorsAttention_result = self.ReLU(x_layer2_SensorsAttention)
        # x_layer2_TimeWinAttention_result = self.ReLU(x_layer2_TimeWinAttention)

        x_layer2_3d = torch.matmul(x_layer1_3d, x_layer2_SensorsAttention_result)
        x_layer2_3d = torch.matmul(x_layer2_3d.permute(0, 2, 1), x_layer2_TimeWinAttention_result).permute(0, 2, 1)

        x_layer2_Res_4d = self.layer2_conv2d_1by1_Res(x_res)
        x_layer2_Res_3d = torch.flatten(x_layer2_Res_4d, start_dim=1, end_dim=2)
        x_layer2_3d = self.BatchNorm(x_layer2_3d + x_layer2_Res_3d)
        x_layer2_3d = self.ReLU(x_layer2_3d)

        return x_layer2_3d

class SegRNN_GA(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(SegRNN_GA, self).__init__()

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
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

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
            self.attention_layer = New_AttentionBlockBranch(self.enc_in, self.seq_len)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1) # b,c,s

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
        enc_out = self.encoder(x_enc) + x_enc
        enc_out = self.attention_layer(enc_out) +enc_out
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.rul_projection(output)
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
