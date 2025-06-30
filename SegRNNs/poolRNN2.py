import torch
import torch.nn as nn
import torch.nn.functional as F

class poolRNN2(nn.Module):
    def __init__(self, configs):
        super(poolRNN2, self).__init__()

        # Extract parameters from configs
        self.seq_len = configs.seq_len  # window_size
        self.enc_in = configs.enc_in  # c (input feature dimension)
        self.d_model = configs.d_model  # model dimension
        self.dropout = configs.dropout
        self.task_name = configs.task_name

        # Additional parameters required for the new model
        self.seg_len = configs.seg_len  # Target sequence length after pooling
        self.num_heads = configs.n_heads  # Number of attention heads

        # Pooling layer to reduce window_size to n
        self.pool = nn.AvgPool1d(kernel_size=self.seg_len, stride=self.seg_len)

        # Linear layer to project input features to d_model
        self.input_proj = nn.Conv1d(in_channels=self.enc_in, out_channels=self.d_model, kernel_size=3, padding=1)

        # GRU layer with input size d_model and hidden size d_model
        self.gru = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True)

        # 1D CNN to transform hidden state from d_model to 2*d_model
        self.cnn = nn.Conv1d(in_channels=self.d_model, out_channels=2 * self.d_model, kernel_size=1)

        # 1D CNN for output
        self.cnn_output = nn.Conv1d(in_channels=self.d_model, out_channels=2 * self.d_model, kernel_size=1)

        # Self-attention layer for hn (kept as original)
        self.self_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1)

        # Self-attention layer for output
        self.self_attn_output = nn.MultiheadAttention(embed_dim=2 * self.d_model, num_heads=self.num_heads)

        # Linear layer for RUL prediction
        self.rul_projection = nn.Linear(self.d_model, 1)

    def encoder(self, x_enc):
        batch_size = x_enc.size(0)
        # Input: x_enc shape (b, window_size, c)

        # Step 1: Pooling on the sequence dimension
        x = x_enc.permute(0, 2, 1)  # (b, c, window_size)
        x_pooled = self.pool(x)  # (b, c, n)
        x_pooled = x_pooled.permute(0, 2, 1)  # (b, n, c)

        # Step 2: Project input features to d_model
        x_pooled = self.input_proj(x_pooled.permute(0, 2, 1)).permute(0, 2, 1)  # (b, n, d_model)

        # Step 3: First GRU pass
        output, hn = self.gru(x_pooled)  # output: (b, n, d_model), hn: (1, b, d_model)

        # Step 4: Process hn (same as original)
        hn_cnn = self.cnn(hn.permute(1, 2, 0))  # (b, 2*d_model, 1)
        hn_cnn = hn_cnn.permute(1, 0, 2)  # (2*d_model, b, 1)
        attn_output, _ = self.self_attn(hn_cnn, hn_cnn, hn_cnn)  # (2*d_model, b, 1)
        attn_output = attn_output.permute(2, 1, 0)  # (1, b, 2*d_model)
        new_hn = attn_output[:, :, self.d_model:].contiguous()  # (1, b, d_model)

        # Step 5: Process output
        output_cnn = self.cnn_output(output.permute(0, 2, 1)).permute(0, 2, 1)  # (b, n, 2*d_model)
        output_attn, _ = self.self_attn_output(
            output_cnn.permute(1, 0, 2),  # query: (n, b, 2*d_model)
            output_cnn.permute(1, 0, 2),  # key: (n, b, 2*d_model)
            output_cnn.permute(1, 0, 2)   # value: (n, b, 2*d_model)
        )  # (n, b, 2*d_model)
        new_output = output_attn[:, :, self.d_model:].permute(1, 0, 2)  # (b, n, d_model)

        # Step 6: Second GRU pass
        output2, hn2 = self.gru(new_output, new_hn)  # output2: (b, n, d_model), hn2: (1, b, d_model)

        # Return the final hidden state squeezed
        return hn2.squeeze(0)  # (b, d_model)

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