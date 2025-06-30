import torch
import torch.nn as nn
import torch.nn.functional as F

class SegRNN_CNN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(SegRNN_CNN, self).__init__()

        # 获取参数
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers
        self.task_name = configs.task_name
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.seq_len // self.seg_len
        self.num_filters = getattr(configs, 'num_filters', 64)  # 默认值 64
        self.kernel_size = getattr(configs, 'kernel_size', 3)   # 默认值 3

        # 构建模型
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

        # CNN 层用于 RUL 预测
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.enc_in, self.num_filters, self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(self.num_filters, self.num_filters * 2, self.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 计算卷积后的展平大小
        self.flat_size = self.num_filters * 2 * (self.seq_len // 4)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )

        # 分类任务的投影层
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.enc_in * self.seq_len, configs.num_class)

    def encoder(self, x):
        batch_size = x.size(0)

        # 归一化和维度调整
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        # 分段和嵌入
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))  # (batch_size * enc_in, seg_num_x, d_model)

        # 位置嵌入
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)  # (batch_size * enc_in, 1, d_model)

        # 编码
        _, hn = self.rnn(x)  # (1, batch_size * enc_in, d_model)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        for i in range(self.num_layers - 1):
            _, hy = self.rnn(pos_emb, hy)

        # 预测
        y = self.predict(hy).view(-1, self.enc_in, self.seq_len)
        y = y.permute(0, 2, 1) + seq_last  # (batch_size, seq_len, enc_in)
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
        # 编码
        enc_out = self.encoder(x_enc) + x_enc  # (batch_size, seq_len, enc_in)

        # CNN 处理
        conv_out = self.conv_layers(enc_out.permute(0, 2, 1))  # (batch_size, num_filters * 2, seq_len // 4)

        # 展平
        flat_out = conv_out.reshape(conv_out.shape[0], -1)  # (batch_size, num_filters * 2 * (seq_len // 4))

        # 全连接层
        output = self.fc_layers(flat_out)  # (batch_size, 1)
        return output

    def forward(self, x_enc):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.seq_len:, :]  # [batch_size, seq_len, enc_in]
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