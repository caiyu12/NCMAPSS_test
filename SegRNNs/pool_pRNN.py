import torch
import torch.nn as nn
import torch.nn.functional as F


class pool_pRNN(nn.Module):
    def __init__(self, configs):
        super(pool_pRNN, self).__init__()

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
        self.gru2 = nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=True)

        # 1D CNN to transform hidden state from d_model to 2*d_model
        self.cnn = nn.Conv1d(in_channels=self.d_model, out_channels=2 * self.d_model, kernel_size=3, padding=1)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1)
        self.pos_emb = nn.Parameter(torch.randn(1, self.d_model))
        self.dsqueeze = nn.Linear(64, 1)

        # Linear layer for RUL prediction
        self.rul_projection = nn.Linear(self.d_model, 1)

    def encoder(self, x_enc):
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
        hn_cnn = hn_cnn.permute(1, 0, 2)  # (2*d_model, b, 1)

        # Step 5: Self-attention on the hidden state
        attn_output, _ = self.self_attn(hn_cnn, hn_cnn, hn_cnn)  # (2*d_model, b, 1)
        attn_output = attn_output.permute(2,1,0) # (1, b, 2*d_model)

        # Step 6: Extract latter half of attention output
        new_hn = attn_output[:, :, self.d_model:].contiguous()  # (1, b, d_model)

        pos_emb = self.pos_emb.unsqueeze(0).permute(2,1,0).repeat(batch_size, 1, 1) #(bd,1,1)
        _, hn2 = self.gru2(pos_emb, new_hn.permute(1,2,0).reshape(1, -1, 1)) # bd,1,1  1,bd,d'
        hn2 = hn2.squeeze(2).reshape(-1, self.d_model)

        # Step 7: Second GRU pass using original output as input and new hidden state
        # output2, hn2 = self.gru(output, new_hn)  # output2: (b, s, d_model), hn2: (1, b, d_model)

        # Return the final hidden state squeezed
        # return hn2.squeeze(0)  # (b, d_model)
        return hn2

    def rul_prediction(self, x_enc):
        # Encode the input
        enc_out = self.encoder(x_enc)  # (b, d_model)

        # Project to RUL prediction
        output = self.rul_projection(enc_out)  # (b, 1)
        return output

    def forward(self, x_enc):
        if self.task_name == 'rul_prediction':
            return self.rul_prediction(x_enc)
        # Placeholder for other tasks if needed
        return None

# Example usage:
# configs = type('Configs', (), {'seq_len': 100, 'enc_in': 14, 'd_model': 64, 'dropout': 0.1, 's': 10, 'num_heads': 8, 'task_name': 'rul_prediction'})()
# model = pRNN(configs)
# x_enc = torch.randn(32, 100, 14)  # (b, window_size, c)
# output = model(x_enc)  # (b, 1)