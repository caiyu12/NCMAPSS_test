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

class pRNN_sq(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(pRNN_sq, self).__init__()

        # Get parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers

        self.task_name = configs.task_name
        if self.task_name in ['classification', 'anomaly_detection', 'imputation', 'rul_prediction']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len

        # Building model
        self.channel = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.d_model, 4),
            # nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.squeeze_layer = nn.Linear(self.enc_in, 4)  # Added to squeeze channels to 4
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(4, self.d_model // 2))  # Changed to 4

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Task-specific layers
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(4 * configs.seq_len, configs.num_class)  # Adjusted to 4
        elif self.task_name == 'rul_prediction':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.rul_projection = nn.Linear(4 * configs.seq_len, 1)  # Adjusted to 4
            # self.rul_head = AttentionPoolingRULHead(self.seq_len, self.enc_in, configs.dropout)
        elif self.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection']:
            self.output_projection = nn.Linear(4, self.enc_in)  # Added for tasks needing enc_in channels

    def encoder(self, x):
        # Input shape: (batch_size, seq_len, enc_in)
        batch_size = x.size(0)

        # Squeeze channel dimension to 4
        x = self.channel(x)  # (batch_size, seq_len, 4)

        # Normalization and permute
        seq_last = x[:, -1:, :].detach()  # (batch_size, 1, 4)
        x = (x - seq_last).permute(0, 2, 1)  # (batch_size, 4, seq_len)

        # Embedding
        x = x.reshape(batch_size * 4, self.seg_num_x, self.seg_len)  # (batch_size * 4, seg_num_x, seg_len)
        x = self.valueEmbedding(x)  # (batch_size * 4, seg_num_x, d_model)

        # Encoding
        _, hn = self.rnn(x)  # hn: (1, batch_size * 4, d_model)

        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(4, 1, 1),  # (4, 1, d_model // 2)
            self.channel_emb.unsqueeze(1)  # (4, 1, d_model // 2)
        ], dim=-1)  # (4, 1, d_model)
        pos_emb = pos_emb.view(-1, 1, self.d_model).repeat(batch_size, 1, 1)  # (batch_size * 4, 1, d_model)

        _, hy = self.rnn(pos_emb, hn.view(1, -1, self.d_model))  # (batch_size * 4, 1, d_model)
        for i in range(self.num_layers - 1):
            _, hy = self.rnn(pos_emb, hy)

        # Predict
        y = self.predict(hy).view(batch_size, 4, self.pred_len)  # (batch_size, 4, pred_len)
        y = y.permute(0, 2, 1) + seq_last  # (batch_size, pred_len, 4)
        return y

    def forecast(self, x_enc):
        enc_out = self.encoder(x_enc)  # (batch_size, pred_len, 4)
        if hasattr(self, 'output_projection'):
            enc_out = self.output_projection(enc_out)  # (batch_size, pred_len, enc_in)
        return enc_out

    def imputation(self, x_enc):
        enc_out = self.encoder(x_enc)  # (batch_size, seq_len, 4)
        if hasattr(self, 'output_projection'):
            enc_out = self.output_projection(enc_out)  # (batch_size, seq_len, enc_in)
        return enc_out

    def anomaly_detection(self, x_enc):
        enc_out = self.encoder(x_enc)  # (batch_size, seq_len, 4)
        if hasattr(self, 'output_projection'):
            enc_out = self.output_projection(enc_out)  # (batch_size, seq_len, enc_in)
        return enc_out

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)  # (batch_size, seq_len, 4)
        output = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, seq_len * 4)
        output = self.projection(output)  # (batch_size, num_class)
        return output

    def rul_prediction(self, x_enc):
        enc_out = self.encoder(x_enc) + self.channel(x_enc) # (batch_size, seq_len, 4)
        output = enc_out.reshape(enc_out.shape[0], -1)  # (batch_size, seq_len * 4)
        output = self.rul_projection(output)  # (batch_size, 1)
        return output

    def forward(self, x_enc):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [batch_size, pred_len, enc_in]
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [batch_size, seq_len, enc_in]
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [batch_size, seq_len, enc_in]
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [batch_size, num_class]
        elif self.task_name == 'rul_prediction':
            dec_out = self.rul_prediction(x_enc)
            return dec_out  # [batch_size, 1]
        return None