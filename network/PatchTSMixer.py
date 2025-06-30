import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePatchTSMixer(nn.Module):
    def __init__(self, configs):
        super(SimplePatchTSMixer, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in  # num_channels
        self.d_model = configs.d_model
        self.seg_len = configs.seg_len
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout

        # Calculate number of patches
        self.num_patches = self.seq_len // self.seg_len

        # Feature projection
        self.feature_projection = nn.Linear(self.seg_len, self.d_model)

        # Mixer layers
        self.mixer_layers = nn.ModuleList([MixerLayer(self.d_model, self.num_patches, self.dropout) for _ in range(self.num_layers)])

        # RUL prediction head
        self.rul_head = nn.Linear(self.enc_in * self.num_patches * self.d_model, 1)

    def forward(self, x):
        # x: [b, s, c]
        x = x.permute(0, 2, 1)
        b, c, s = x.shape

        # Unfold to patches: [b, c, s] -> [b, c, n, seg_len]
        x = x.unfold(-1, self.seg_len, self.seg_len).permute(0, 2, 1, 3)  # [b, n, c, seg_len]

        # Feature projection: [b, n, c, seg_len] -> [b, n, c, d_model]
        x = self.feature_projection(x)  # [b, n, c, d_model]

        # Apply mixer layers
        for layer in self.mixer_layers:
            x = layer(x)  # [b, n, c, d_model]

        # Reshape back to [b, s', c], assuming s' = n * d_model
        x = x.permute(0, 2, 1, 3).reshape(b, c, self.num_patches * self.d_model)  # [b, c, s']
        x = x.reshape(b, -1)  # [b, c * s']

        # RUL prediction
        rul_output = self.rul_head(x)  # [b, 1]
        return rul_output

class MixerLayer(nn.Module):
    def __init__(self, d_model, num_patches, dropout):
        super(MixerLayer, self).__init__()
        self.patch_mixer = PatchMixer(d_model, num_patches, dropout)
        self.feature_mixer = FeatureMixer(d_model, dropout)

    def forward(self, x):
        # x: [b, n, c, d_model]
        x = self.patch_mixer(x)  # Mix patches
        x = self.feature_mixer(x)  # Mix features
        return x

class PatchMixer(nn.Module):
    def __init__(self, d_model, num_patches, dropout):
        super(PatchMixer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_patches, num_patches * 2),
            nn.GELU(),
            nn.Linear(num_patches * 2, num_patches),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [b, n, c, d_model]
        # Transpose to [b, c, d_model, n]
        x = x.permute(0, 2, 3, 1)  # [b, c, d_model, n]
        # Apply MLP on patch dimension
        x = self.mlp(x)  # [b, c, d_model, n]
        # Transpose back to [b, n, c, d_model]
        x = x.permute(0, 3, 1, 2)
        return x

class FeatureMixer(nn.Module):
    def __init__(self, d_model, dropout):
        super(FeatureMixer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [b, n, c, d_model]
        # Apply MLP on feature dimension
        x = self.mlp(x)  # [b, n, c, d_model]
        return x
