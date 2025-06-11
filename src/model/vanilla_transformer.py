import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayerRMSNorm(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, Freq, d]

        # --- Self-Attention Block ---
        attn_output, _ = self.self_attn(x, x, x)  # [B, Freq, d]
        x = x + self.dropout(attn_output)         # Residual connection
        x = self.norm1(x)                         # RMSNorm

        # --- Feedforward Block ---
        ff_output = self.linear2(self.activation(self.linear1(x)))  # [B, Freq, d]
        x = x + self.dropout(ff_output)          # Residual connection
        x = self.norm2(x)                        # RMSNorm

        return x

class VanillaTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayerRMSNorm(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(embedding_dim)

    def forward(self, x):
        # x: [B, Freq, d]
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

""""
VanillaTransformerEncoder/
│
├── Input:
│   └── x_raw: Tensor [B, Freq, d]                     # Sequence of connectome tokens (from GNN)
│   ├── freq_bounds: Tensor [Freq, 2]                  # EEG band bounds [f_l, f_u]
│   ├── age: Tensor [B, 1]
│   └── gender: Tensor [B, 1]
│
├── Step 1 — Demographic Rotary Frequency Encoding (dRoFE):
│   └── x = dRoFEEmbedding(x_raw, freq_bounds, age, gender)
│       → encodes band-specific phase + age-dependent magnitude + gender bias
│       → produces x_rot ∈ R^{B * Freq * d}
│
├── Step 2 — Layers (repeated L times):
│   ├── Multi-Head Self-Attention:
│   │   ├── Q, K, V = Linear(x)
│   │   ├── AttnScore = Q · Kᵀ / √d
│   │   ├── AttnWeights = softmax(AttnScore)
│   │   └── AttnOut = AttnWeights · V
│   ├── Residual Connection
│   ├── RMSNorm
│   ├── Feed Forward Network (GELU activation):
│   │   └── FF = Linear(GELU(Linear(x)))
│   ├── Residual Connection
│   └── RMSNorm
│
└── Output:
    └── x_out: Tensor [B, Freq, d]                     # Coarse token representations
"""