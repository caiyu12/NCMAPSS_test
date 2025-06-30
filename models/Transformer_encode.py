import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class mymodel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.enc_in = config.enc_in
        self.d_model = config.d_model
        self.max_len = config.max_len
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=config.dropout, max_len=config.max_len)
        encoder_layers = TransformerEncoderLayer(self.d_model, config.nhead, 512, dropout=config.dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.nlayers)
        self.dropout = nn.Dropout(config.dropout)
        self.decoder = nn.Linear(self.d_model, 1)
        self.init_weights()
        self.embed = nn.Linear(self.enc_in, self.d_model)
        self.squeeze = nn.Linear(config.seq_len, 1)


    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, key_msk, x_dec, x_mark_dec, attn_msk=None) -> Tensor:
        """
        return:
            output1: Tensor, extracted features
            output2: Tensor, predicted series
        """
        src = self.embed(src)
        src = self.pos_encoder(src)
        output1 = self.transformer_encoder(src, attn_msk, None)
        output1 = self.dropout(output1)
        output2 = self.decoder(output1)
        enc_out = output2.reshape(output2.shape[0], -1)
        output2 = enc_out.mean(dim=1, keepdim=True)
        # output2 = self.squeeze(enc_out)
        return output2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, feature_num]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)