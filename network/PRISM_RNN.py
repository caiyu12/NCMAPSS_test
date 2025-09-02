import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import SmoothedAnchorNorm, UnifiedGMMGate, series_decomp, FreMLPChannel, FreCNNChannel, DynamicInput

#PRISM(Prognostic Routing of Independent Strategy Modules)
class PRISM_RNN(nn.Module):
    def __init__(self, configs):
        super(PRISM_RNN, self).__init__()
        self.name = 'Cluster_RNN'
        self.layer = configs.e_layers
        self.accept_window = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.seq_len = configs.seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.pred_len = 1
        self.dropout = configs.dropout
        self.lstm_layer_num = 1
        self.register_buffer('gate_value', None)
        self.cluster_map = None
        self.batch_size = None
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, self.d_model))
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.valueSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.seg_len),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )
        self.SensorsEmbedding_shifted = nn.Sequential(
            DynamicInput(self.d_model),
            nn.ReLU()
        )
        self.SensorsEmbedding = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU()
        )
        self.SensorSqueeze = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in),
            nn.ReLU(),
            # nn.Dropout(self.dropout)
        )
        self.dynamic_squeeze_layers = nn.ModuleDict()
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.norm = nn.BatchNorm1d(self.seq_len)
        self.projection = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze = nn.Sequential(
            nn.Linear(self.enc_in, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.robust_norm = SmoothedAnchorNorm()
        # self.cluster_map_generator = InterpretableClusteringGate()
        self.cluster_gate = UnifiedGMMGate()
        # self.DBSCAN_gate = InterpretableClusteringGate()
        self.rnn_cd = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num,
                          batch_first=True)
        self.rnn_ci = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layer_num,
                          batch_first=True)
        self.channel = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.d_model, self.enc_in),
            # nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.seg_num_x, self.seq_len)
        )
        self.FreMLP = FreMLPChannel(embed_size=32, feature_size=self.enc_in, seq_len=self.seq_len)
        self.FreCNN = FreCNNChannel(feature_size=self.enc_in, seq_len=self.seq_len)
        self.SeriesDecomp = series_decomp(kernel_size=5)

    def forecast_cd_shifted(self, x_enc, channel_size):
        """
        使用通道依赖策略的RNN，将长序列分为多个子序列并行处理，
        由此减轻因序列过长导致的RNN遗忘问题。
        input:b,s,c; output:b,s,c
        """
        batch_size, _, channels = x_enc.shape
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, channel_size).permute(0, 2, 1, 3)
        x = self.SensorsEmbedding_shifted(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d
        x = self.rnn_cd(x)[0]  # bs,n,d
        key = str(channel_size)
        if key not in self.dynamic_squeeze_layers:
            self.dynamic_squeeze_layers[key] = nn.Linear(self.d_model, channel_size).to(x_enc.device)
        x = self.dynamic_squeeze_layers[key](x)
        x = F.relu(x).reshape(batch_size, -1, channel_size)  # 手动应用ReLU
        # self.SensorSqueeze[0] = nn.Linear(self.d_model, channels).to(x_enc.device)
        # x = self.SensorSqueeze(x).reshape(batch_size, -1, channel_size)
        x = self.norm(x)  # b,s,c
        return x

    def forecast_cd(self, x_enc):
        """
        使用通道依赖策略的RNN，将长序列分为多个子序列并行处理，
        由此减轻因序列过长导致的RNN遗忘问题。
        input:b,s,c; output:b,s,c
        """
        batch_size, _, _ = x_enc.shape
        x_enc = self.FreMLP(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        # Reshape into batch, seg_num, seg_len, channel (enc_in)
        #   and permute into batch, seg_len, seg_num, channel (enc_in)
        x_enc = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3)

        # Embed channel into d_model
        #   and reshape into batch * seg_len, seg_num, d_model
        x = self.SensorsEmbedding(x_enc).reshape(-1, self.seg_num_x, self.d_model)  # bs,n,d

        # Traditional batch-first RNN accepts tensors of shape (batch, seq_len, feature_size).
        # Here, tensors of different relative positions in the segmented sequences are treated as
        #   different samples, combined in the batch dimension to capture channel-dependent information. Each
        #   input contains embedded channel features across all segments at the same relative position.
        # The seq_num is treated as the seq_len for RNN processing.
        # Example:
        #   Sequence:
        #       [t1, t2, t3, t4, t5, t6, t7, t8] (seq_len=8), choosing seg_len=2, we have seg_num=4
        # -> Segmented sequences:
        #       [[t1, t2], [t3, t4], [t5, t6], [t7, t8]]
        # -> Permuted:
        #       [[t1, t3, t5, t7], [t2, t4, t6, t8]]
        # -> RNN inputs:
        #     1: [t1, t3, t5, t7]
        #     2: [t2, t4, t6, t8]
        # Look back into their original position in the sequence, you can find the perception field of RNN
        #   covers the entire sequence, while each input only contains partial information of the sequence. This
        #   helps alleviate the forgetting problem of RNN on long sequences.
        x, hn = self.rnn_cd(x) # x:bs,n,d  hn:1,bs,d
        x = self.SensorSqueeze(x).reshape(batch_size, -1, self.enc_in) #b,s,c
        x = self.norm(x) # b,s,c
        return x

    def forecast_ci(self, x_enc):
        """
        使用通道独立策略的RNN，将通道维度并到batch维度从而实现并行训练。
        input:b,s,c; output:b,s,c
        """
        batch_size, _, _ = x_enc.shape
        # x_enc, seq_last = self.robust_norm(x_enc)
        x = x_enc.permute(0, 2, 1)  # b,c,s
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len)) #bc,n,d
        x, hn = self.rnn_ci(x) #bc,n,d  1,bc,d
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.d_model) #b,c,n,d
        x = self.valueSqueeze(x).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1) #b,w,c
        x = self.channel(x)
        return x

    def forward(self, x_enc):
        # x_enc: [b, s, c]
        batch_size = x_enc.shape[0]
        self.batch_size = batch_size
        if self.gate_value is None or self.gate_value.device != x_enc.device:
            gate, cluster_map = self.cluster_gate(x_enc)
            self.gate_value = torch.tensor([gate], device=x_enc.device)
            self.cluster_map = cluster_map  # 缓存cluster_map以供使用
        else:
            gate = self.gate_value.item()
            cluster_map = self.cluster_map

        if gate == 0:
            x = self.forecast_cd(x_enc)
        else:
            processed_x = torch.zeros_like(x_enc)
            seasonal_x = torch.zeros_like(x_enc)
            # 识别主群体和离群群体
            main_channels = cluster_map.get("main_cluster", [])
            outlier_channels = cluster_map.get("outlier_cluster", [])
            x_enc, seq_last = self.robust_norm(x_enc)
            # 对主群体进行增强 (使用通道依赖路径)
            x_outlier_group = x_enc[:, :, outlier_channels]
            outlier_seasonal, outlier_trend = self.SeriesDecomp(x_outlier_group)
            seasonal_x[:, :, outlier_channels] = outlier_seasonal
            seasonal_x[:, :, main_channels] = x_enc[:, :, main_channels]
            seasonal_output = self.forecast_cd(seasonal_x)
            processed_x[:, :, outlier_channels] = outlier_trend
            # seasonal_output = self.forecast_ci(seasonal_output) + seq_last
            # trend_output = self.forecast_ci(processed_x) + seq_last
            # x = trend_output + seasonal_output
            x = seasonal_output + processed_x
            x = self.forecast_ci(x) + seq_last

        enc_out = self.projection(x.transpose(1, 2)).transpose(1, 2)
        enc_out_2d = enc_out.view(-1, self.enc_in)
        enc_output = self.squeeze(enc_out_2d)

        return enc_output






