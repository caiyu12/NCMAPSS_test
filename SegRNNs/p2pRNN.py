import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        pos_emb = self.pe[:x.size(1), :].unsqueeze(0)
        return pos_emb

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


class p2pRNN(nn.Module):
    def __init__(self, configs):
        super(p2pRNN, self).__init__()

        # Extract parameters from configs
        self.seq_len = configs.seq_len  # window_size
        self.enc_in = configs.enc_in  # c (input feature dimension)
        self.d_model = configs.d_model  # model dimension
        self.dropout = configs.dropout
        self.task_name = configs.task_name

        # Additional parameters required for the new model
        self.seg_len = configs.seg_len  # Target sequence length after pooling
        self.num_heads = configs.n_heads  # Number of attention heads

        # Pooling layer to reduce window_size to s
        self.pool = nn.AvgPool1d(kernel_size=self.seg_len, stride=self.seg_len)

        # Linear layer to project input features to d_model
        self.input_proj = nn.Conv1d(in_channels=self.enc_in, out_channels=self.d_model, kernel_size=3, padding=1)

        # GRU layer with input size d_model and hidden size d_model
        self.gru = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=self.seq_len//self.seg_len, hidden_size=self.seq_len//self.seg_len, num_layers=1, batch_first=True)

        # 1D CNN to transform hidden state from d_model to 2*d_model
        self.cnn = nn.Conv1d(in_channels=self.d_model, out_channels=2 * self.d_model, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model, kernel_size=3 , padding=1)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_len, self.seq_len//self.seg_len))
        self.dsqueeze = nn.Linear(64, 1)
        self.channel_attn_head = AttentionPoolingRULHead(self.seq_len, self.d_model)

        # Linear layer for RUL prediction
        self.rul_projection = nn.Linear(self.d_model, 1)

    def encoder(self, x_enc):
        batch_size = x_enc.size(0)
        # Input: x_enc shape (b, window_size, c)

        # Step 1: Pooling on the sequence dimension
        # (b, window_size, c) -> (b, c, window_size) for pooling
        x_pooled = x_enc.reshape(batch_size, self.seg_num_x, self.seg_len, self.enc_in).permute(0, 2, 1, 3) #b,s,n,c
        x_pooled = x_pooled.reshape(-1, self.seg_num_x, self.enc_in)

        # Step 2: Project input features to d_model
        x_pooled = self.input_proj(x_pooled.permute(0, 2, 1)).permute(0, 2, 1)  # (bs, n, d_model)

        # Step 3: First GRU pass
        output, hn = self.gru(x_pooled)  # output: (bs, n, d_model), hn: (1, bs, d_model)

        # Step 4: Apply 1D CNN to hidden state
        # (1, b, d_model) -> (b, d_model, 1) for Conv1d
        hn_cnn = self.cnn(hn.permute(1, 2, 0))  # (bs, 2*d_model, 1)
        new_hn = self.cnn2(hn_cnn).permute(2, 0, 1) #(1,bs,2d)

        #attention on channel dimension
        # hn_cnn = hn_cnn.permute(1, 0, 2)  # (2*d_model, b, 1)
        # attn_output, _ = self.self_attn(hn_cnn, hn_cnn, hn_cnn)  # (2*d_model, b, 1)
        # attn_output = attn_output.permute(2,1,0) # (1, b, 2*d_model)
        # new_hn = attn_output[:, :, self.d_model:].contiguous()  # (1, b, d_model)

        output = output.permute(0,2,1).reshape(-1,1, self.seq_len//self.seg_len) #bsd,1,n
        pos_emb = self.pos_emb.unsqueeze(0).repeat(self.d_model, 1, 1).repeat(batch_size,1,1) #bsd,1,n
        output,_ = self.gru2(output,pos_emb.permute(1,0,2)) #bsd,1,n    1,bsd,n
        output = output.reshape(batch_size*self.seg_len, self.d_model, -1).permute(0,2,1) #bs,n,d

        # Step 7: Second GRU pass using original output as input and new hidden state
        output2, hn2 = self.gru(output, new_hn)  # output2: (bs, n, d_model), hn2: (1, bs, d_model)

        # Return the final hidden state squeezed
        return output2

    def forecast(self, x_enc):
        batch_size = x_enc.size(0)
        # Input: x_enc shape (b, window_size, c)

        # Step 1: Pooling on the sequence dimension
        # (b, window_size, c) -> (b, c, window_size) for pooling
        x = x_enc.permute(0, 2, 1)
        x_pooled = self.pool(x)  # (b, c, n)
        x_pooled = x_pooled.permute(0, 2, 1)  # (b, n, c)

        # Step 2: Project input features to d_model
        x_pooled = self.input_proj(x_pooled.permute(0, 2, 1)).permute(0, 2, 1)  # (b, n, d_model)

        # Step 3: First GRU pass
        output, hn = self.gru(x_pooled)  # output: (b, n, d_model), hn: (1, b, d_model)

        # Step 4: Apply 1D CNN to hidden state
        # (1, b, d_model) -> (b, d_model, 1) for Conv1d
        hn_cnn = self.cnn(hn.permute(1, 2, 0))  # (b, 2*d_model, 1)
        new_hn = self.cnn2(hn_cnn).permute(2, 0, 1)

        #attention on channel dimension
        # hn_cnn = hn_cnn.permute(1, 0, 2)  # (2*d_model, b, 1)
        # attn_output, _ = self.self_attn(hn_cnn, hn_cnn, hn_cnn)  # (2*d_model, b, 1)
        # attn_output = attn_output.permute(2,1,0) # (1, b, 2*d_model)
        # new_hn = attn_output[:, :, self.d_model:].contiguous()  # (1, b, d_model)

        output = output.permute(0,2,1).reshape(-1,1, self.seq_len//self.seg_len) #bd,1,n
        pos_emb = self.pos_emb.unsqueeze(0).repeat(self.d_model, 1, 1).repeat(batch_size,1,1) #bd,1,n
        output,_ = self.gru2(output,pos_emb.permute(1,0,2)) #bd,1,n    1,bd,n
        output = output.reshape(batch_size, self.d_model, -1).permute(0,2,1)

        # Step 7: Second GRU pass using original output as input and new hidden state
        output2, hn2 = self.gru(output, new_hn)  # output2: (b, s, d_model), hn2: (1, b, d_model)

        # Return the final hidden state squeezed
        return output2

    def rul_prediction(self, x_enc):
        # Encode the input
        enc_out = self.encoder(x_enc) # (bs, n, d)
        output = self.channel_attn_head(enc_out.reshape(-1, self.seq_len, self.d_model)) #(b,s,1)

        # Project to RUL prediction
        # output = self.rul_projection(enc_out)  # (b, 1)
        return output

    def forward(self, x_enc):
        if self.task_name == 'rul_prediction':
            return self.rul_prediction(x_enc)
        # Placeholder for other tasks if needed
        return None