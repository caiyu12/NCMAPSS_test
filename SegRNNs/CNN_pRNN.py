import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [batch_size, seq_len, enc_in]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x

class CNN_pRNN(nn.Module):
    def __init__(self, configs):
        super(CNN_pRNN, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.decoder_layers
        self.layer = configs.e_layers
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len

        # Value Embedding for encoder
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        # CNN 特征提取层
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(self.enc_in, configs.CNN_size, kernel_size=configs.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 池化到 1
            nn.Conv1d(configs.CNN_size, 2*configs.CNN_size, kernel_size=configs.kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cnn_encode = nn.Linear(self.seq_len//4, configs.d_model)
        self.cnn_squeeze =  nn.Linear(2*configs.CNN_size, self.enc_in)

        # pRNN 编码器（ResBlock + GRU）
        self.TSMixerBlock = nn.ModuleList([ResBlock(configs) for _ in range(configs.e_layers)])
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1,
                                  batch_first=True, bidirectional=False)

        # 位置嵌入和通道嵌入
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        # RUL 预测头
        self.rul_projection = nn.Linear(self.d_model, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, enc_in]
        batch_size = x.size(0)

        # 归一化
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last  # [batch_size, seq_len, enc_in]

        # ResBlock 处理
        for i in range(self.layer):
            x = self.TSMixerBlock[i](x)  # [batch_size, seq_len, enc_in]
        x = x.permute(0, 2, 1)  # [batch_size, enc_in, seq_len]

        x_cnn = self.cnn_layers(x) #b,2*nf,s//4
        x_cnn = self.cnn_squeeze(x_cnn.permute(0, 2, 1)).permute(0, 2, 1) #b,enc_in,s//4
        x_cnn = self.cnn_encode(x_cnn) #b,enc_in,d
        x_cnn_emb = x_cnn.reshape(-1,1,self.d_model) #bc,1,d

        # 分段并嵌入
        x = x.reshape(-1, self.seg_num_x, self.seg_len)  # [batch_size * enc_in, seg_num_x, seg_len]
        x_emb = self.valueEmbedding(x)  # [batch_size * enc_in, seg_num_x, d_model]

        # pRNN 编码器
        _, hn = self.rnn(x_emb)  # hn: [1, batch_size * enc_in, d_model]

        # 位置嵌入
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),  # [enc_in, 1, d_model//2]
            self.channel_emb.unsqueeze(1)  # [enc_in, 1, d_model//2]
        ], dim=-1)  # [enc_in, 1, d_model]
        pos_emb = pos_emb.view(-1, 1, self.d_model).repeat(batch_size, 1, 1)  # [batch_size * enc_in, 1, d_model]

        # 解码器输入：CNN 输出 + 位置嵌入
        decoder_input =pos_emb  # [batch_size * enc_in, 1, d_model]

        # pRNN 解码器
        _, hy = self.rnn(decoder_input, hn)  # hy: [1, batch_size * enc_in, d_model]
        for i in range(self.num_layers - 1):
            _, hy = self.rnn(decoder_input, hy)
        hy = hy + x_cnn_emb.permute(1,0,2)

        # RUL 预测
        rul_output = self.rul_projection(hy.squeeze(0))  # [batch_size * enc_in, 1]
        rul_output = rul_output.view(batch_size, self.enc_in, 1).mean(dim=1)  # [batch_size, 1]

        return rul_output