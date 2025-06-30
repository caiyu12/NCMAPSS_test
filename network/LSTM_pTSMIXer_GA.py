import torch.nn as nn
import torch

'''
params：seq_len:input sequence length
pred_len:prediction sequence length
d_model:embedding dimension
n_heads:head number
d_ff:feed forward dimension
dropout:dropout rate
'''
# class SENet(nn.Module):
#     def __init__(self, sensors, seq_len):
#         super().__init__()
#         '''
#         /*--------layer-1---------------------------------------------------------------------*/
#         '''
#         self.layer1_conv2d_1by1 = nn.Conv2d(
#             in_channels=1, out_channels=1,
#             kernel_size=1, stride=1, padding=0,
#             bias=False
#         )
#         '''
#         /*--------layer-2---------------------------------------------------------------------*/
#         '''
#         self.layer2_conv2d_1by1_Res = nn.Conv2d()

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

# class AttentionBlockBranch(nn.Module):
#     def __init__(self, seq_len):
#         super().__init__()
#         '''
#         /*--------layer-1---------------------------------------------------------------------*/
#         '''
#         self.layer1_conv2d_1by1 = nn.Conv2d(
#             in_channels=1, out_channels=1,
#             kernel_size=1, stride=1, padding=0,
#             bias=False
#         )
#         '''
#         /*--------layer-2---------------------------------------------------------------------*/
#         '''
#         self.layer2_conv2d_1by1_Res = nn.Conv2d(
#             in_channels=1, out_channels=1,
#             kernel_size=1, stride=1, padding=0,
#             bias=True
#         )
#         self.layer2_pool_Sensors = nn.MaxPool1d(kernel_size=seq_len, stride=1) # Windows Size
#         self.layer2_pool_TimeWin = nn.AvgPool1d(kernel_size=14, stride=1) # Effective Sensors
#         self.layer2_linear_seq = nn.Sequential(
#             nn.Linear(in_features=seq_len + 14   , out_features=int((seq_len+14)/2)), # 22 = (seq_len + 14)/2
#             nn.Linear(in_features=int((seq_len+14)/2), out_features=seq_len + 14)
#         )
#         '''
#         /*--------activation---------------------------------------------------------------------*/
#         '''
#         self.Sigmoid = nn.Sigmoid()
#         self.TanH = nn.Tanh()
#         self.ReLU = nn.ReLU()
#         '''
#         /*-----------------------------------------------------------------------------*/
#         '''
#         self.BatchNorm = nn.BatchNorm1d(seq_len)
#
#     def forward(self, x : torch.tensor) -> torch.tensor:
#         x_layer1 = self.layer1_conv2d_1by1(x.unsqueeze(dim=1))
#         x_layer1_3d = torch.flatten(x_layer1, start_dim=1, end_dim=2)
#
#         x_layer2_SensorsAttention = self.layer2_pool_Sensors(x_layer1_3d.permute(0, 2, 1)) #(N, Sensors, 1)
#         x_layer2_TimeWinAttention = self.layer2_pool_TimeWin(x_layer1_3d)                  #(N, TimeWin, 1)
#         shape_Sensors = x_layer2_SensorsAttention.shape
#         shape_TimeWin = x_layer2_TimeWinAttention.shape
#
#         x_layer2_SensorsAttention = torch.flatten(x_layer2_SensorsAttention, start_dim=1) #(N, Sensors)
#         x_layer2_TimeWinAttention = torch.flatten(x_layer2_TimeWinAttention, start_dim=1) #(N, TimeWin)
#         x_layer2_linear = self.layer2_linear_seq(torch.cat([x_layer2_SensorsAttention, x_layer2_TimeWinAttention], dim=1))
#         #(N, Sensors+TimeWin)->(N, Sensors+TimeWin)
#         x_layer2_SensorsAttention = x_layer2_linear[:, :14].reshape(*shape_Sensors)
#         x_layer2_TimeWinAttention = x_layer2_linear[:, 14:].reshape(*shape_TimeWin)
#         x_layer2_SensorsAttention_result = self.Sigmoid(x_layer2_SensorsAttention)
#         x_layer2_TimeWinAttention_result = self.Sigmoid(x_layer2_TimeWinAttention)
#
#         x_layer2_3d = torch.matmul(x_layer1_3d, x_layer2_SensorsAttention_result)
#         x_layer2_3d = torch.matmul(x_layer2_3d.permute(0, 2, 1), x_layer2_TimeWinAttention_result).permute(0, 2, 1)
#
#         x_layer2_Res_4d = self.layer2_conv2d_1by1_Res(x_layer1)
#         x_layer2_Res_3d = torch.flatten(x_layer2_Res_4d, start_dim=1, end_dim=2)
#         x_layer2_3d = self.BatchNorm(x_layer2_3d + x_layer2_Res_3d)
#         x_layer2_3d = self.ReLU(x_layer2_3d)
#
#         return x_layer2_3d

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


# class Mixup(nn.Module):
#     def __init__(self, sensors, e_layers, seq_len, d_model, dropout):
#         super(Mixup, self).__init__()
#         self.mixup = nn.ModuleList(
#             [ResBlock(sensors, seq_len, d_model, dropout)
#              for _ in range(e_layers)]
#         )

class LSTM_pTSMixer_GA(nn.Module):
    def __init__(self, sensors, e_layers, t_model, c_model, lstm_layer_num, seq_len, dropout, accept_window):
        super(LSTM_pTSMixer_GA, self).__init__()
        self.name = 'LSTM_pTSMixer_GA'
        self.layer = e_layers
        self.accept_window = accept_window
        self.sensors = sensors
        self.d_model = c_model *2
        self.seg_len = 10
        self.seq_len = seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=sensors, hidden_size=sensors, num_layers=lstm_layer_num, batch_first=True)
        self.rnn = nn.GRU(input_size=2*c_model, hidden_size=2*c_model, num_layers=lstm_layer_num, batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        self.model = nn.ModuleList(
            [ResBlock(sensors, seq_len, t_model, c_model, dropout)
             for _ in range(e_layers)]
        )
        self.norm = nn.BatchNorm1d(seq_len)

        self.attention_layer = New_AttentionBlockBranch(sensors, seq_len)

        self.pred_len = 1
        # self.projection = nn.Linear(seq_len, pred_len)
        # self.squeeze = nn.Linear(sensors, pred_len)
        self.projection = nn.Sequential(
            nn.Linear(seq_len, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze    = nn.Sequential(
            nn.Linear(sensors, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

    def forecast(self, x_enc):
        batch_size = x_enc.shape[0]
        # x_enc = x_enc.permute(0, 2, 1)
        # x = self.valueEmbedding(x_enc.reshape(-1, self.seg_num_x, self.seg_len))
        # x = self.rnn(x)[0] + x #bc, n, d
        # x = self.predict(x).reshape(batch_size, self.sensors, -1).permute(0, 2, 1)
        x = self.lstm(x_enc)[0] + x_enc

        x_sliced = x[:, -self.accept_window:, :].contiguous()

        # x: [B, L, D]
        for i in range(self.layer):
            x_sliced = self.model[i](x_sliced)

        x_sliced = self.norm(x_sliced)

        x_sliced = self.attention_layer(x_sliced)

        enc_out = self.projection(x_sliced.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.sensors)
        enc_output = self.squeeze(enc_out_2d)
        return enc_output

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:]  # [B, L, D]


