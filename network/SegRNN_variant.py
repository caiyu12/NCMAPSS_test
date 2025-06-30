import torch.nn as nn
import torch

class SegRNN_variant(nn.Module):
    def __init__(self, configs):
        super(SegRNN_variant, self).__init__()
        self.name = 'LSTM_pTSMixer_GA'
        self.layer = configs.e_layers
        self.accept_window = configs.seq_len
        self.sensors = configs.enc_in
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.seq_len = configs.seq_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.dropout = configs.dropout
        self.lstm_layer_num = configs.d_layers
        self.num_layers = configs.decoder_layers
        self.seg_num_y = self.seg_num_x
        self.enc_in = configs.enc_in

        # self.lstm = nn.LSTM(input_size=self.sensors, hidden_size=self.sensors, num_layers=self.lstm_layer_num, batch_first=True)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        self.gate_control = nn.Parameter(torch.randn(1))

        # self.model = nn.ModuleList(
        #     [ResBlock(sensors, seq_len, t_model, c_model, dropout)
        #      for _ in range(e_layers)]
        # )
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
        self.hn_proj = nn.Linear(self.seg_len, self.pred_len)
        self.squeeze    = nn.Sequential(
            nn.Linear(self.sensors, self.pred_len),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.mixed = nn.Sequential(
            nn.Linear(self.enc_in * self.seg_num_x, self.enc_in * self.seg_num_x * 2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.enc_in * self.seg_num_x * 2, self.enc_in * self.seg_num_x),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

    def forecast(self, x_enc):
        batch_size = x_enc.shape[0]
        x_enc = x_enc.permute(0,2,1)
        x = self.valueEmbedding(x_enc.reshape(-1, self.seg_num_x, self.seg_len))
        x = x.reshape(batch_size, self.enc_in * self.seg_num_x, self.d_model)
        x = self.mixed(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(-1, self.seg_num_x, self.d_model)
        x_rnn,hn = self.rnn(x)
        x = x_rnn + x #bc, n, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn(pos_emb, hn.view(1, -1, self.d_model))  # bc,1,d  1,bc,d
        # for i in range(self.num_layers - 1):
        #     _, hy = self.rnn(pos_emb, hy)
        # x,hy = self.rnn(x, hy) # bc,n,d  1,bc,d
        hy = self.predict(hy).reshape(batch_size, self.sensors, -1).permute(0, 2, 1) #b,seg,c
        x = self.predict(x).reshape(batch_size, self.sensors, -1).permute(0, 2, 1)
        # x = self.lstm(x_enc)[0] + x_enc

        x_sliced = x[:, -self.accept_window:, :].contiguous()

        # x: [B, L, D]
        # for i in range(self.layer):
        #     x_sliced = self.model[i](x_sliced)

        x_sliced = self.norm(x_sliced)

        # x_sliced = self.attention_layer(x_sliced)
        enc_out_hn = self.hn_proj(hy.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out = self.projection(x_sliced.transpose(1, 2)).transpose(1, 2)
        enc_out = self.gate_control * enc_out + (1 - self.gate_control) * enc_out_hn
        enc_out_2d = enc_out.view(-1, self.sensors)
        enc_output = self.squeeze(enc_out_2d)
        return enc_output

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:]  # [B, L, D]