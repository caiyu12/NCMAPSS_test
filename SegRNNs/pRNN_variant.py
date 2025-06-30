import torch
import torch.nn as nn
import torch.nn.functional as F

class pRNN_variant(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(pRNN_variant, self).__init__()

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
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1,
                              batch_first=True)
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )
        self.predict_x = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )
        self.projection = nn.Sequential(
            nn.Linear(self.seq_len, 1),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.squeeze = nn.Sequential(
            nn.Linear(self.enc_in, 1),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.norm = nn.BatchNorm1d(self.seq_len)
        self.mixed = nn.Sequential(
            nn.Linear(self.enc_in*self.seg_num_x, self.enc_in*self.seg_num_x*2),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.enc_in*self.seg_num_x*2, self.enc_in*self.seg_num_x),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )

        if self.task_name == 'rul_prediction':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.rul_projection = nn.Linear(
                configs.enc_in * self.pred_len, 1)
            # self.rul_head = AttentionPoolingRULHead(self.seq_len, self.enc_in, configs.dropout)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)
        # seq_last = x[:, -1:, :].detach()
        # x = (x - seq_last).permute(0, 2, 1)  # b,c,s
        x = x.permute(0, 2, 1)

        #embedding    b,c,s -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
        x = x.reshape(batch_size, self.enc_in*self.seg_num_x, self.d_model)
        x = self.mixed(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(-1, self.seg_num_x, self.d_model)

        # encoding
        output, hn = self.rnn(x)  # bc,n,d  1,bc,d
        output = self.predict_x(output).reshape(batch_size, self.enc_in, -1)
        hn = self.predict(hn).reshape(batch_size, self.enc_in, -1)

        output = output.permute(0, 2, 1)
        hn = hn.permute(0, 2, 1)
        return output,hn

    def rul_prediction(self, x_enc):
        # Encoder
        x_out, hn = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        # x_output = x_out.reshape(x_out.shape[0], -1)
        x_out = self.norm(x_out)
        hn = self.norm(hn)
        # hn_output = self.projection(hn.permute(0, 2, 1)).permute(0, 2, 1)
        # output = self.squeeze(hn_output)
        x_output = self.projection(x_out.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.squeeze(x_output)
        # output = self.rul_projection(x_output)
        # output = self.rul_head(enc_out)
        return output

    def forward(self, x_enc):
        if self.task_name == 'rul_prediction':
            dec_out = self.rul_prediction(x_enc)
            return dec_out
        return None
