import torch
import torch.nn as nn
import torch.nn.functional as F

class XAIGuidedTransformerLayer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)

        # GeGLU feedforward
        self.ff_linear1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.ff_linear2 = nn.Linear(embedding_dim, embedding_dim)

    def xai_guided_attention(self, x, q_expl, k_expl):
        """
        x: input sequence (used as Value)
        q_expl, k_expl: refined attention queries and keys
        """
        d = x.size(-1)
        attn_scores = torch.matmul(q_expl, k_expl.transpose(-1, -2)) / (d ** 0.5)  # [B, F, F]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, x)  # [B, F, d]

    def geglu(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # split on last dim
        return F.gelu(x1) * x2

    def forward(self, x, q_expl, k_expl):
        # Step 1: XAI-guided attention
        attn_out = self.xai_guided_attention(x, q_expl, k_expl)
        x = self.norm1(x + self.dropout(attn_out))

        # Step 2: Feedforward (GeGLU)
        ff = self.ff_linear2(self.geglu(self.ff_linear1(x)))
        x = self.norm2(x + self.dropout(ff))

        return x

class XAIGuidedTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            XAIGuidedTransformerLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(embedding_dim)

    def forward(self, x, q_expl, k_expl):
        """
        x: [B, F, d]
        q_expl, k_expl: [B, F, d]
        """
        for layer in self.layers:
            x = layer(x, q_expl, k_expl)
        return self.final_norm(x)


"""
XAIGuidedTransformerEncoder/
│
├── Input:
│   ├── x: Tensor [B, Freq, d]                 # Sequence of token embeddings (from vanilla Transformer or dRoFE)
│   ├── q_expl: Tensor [B, Freq, d]            # XAI-guided refined Query vectors (e.g. DeepLIFT)
│   └── k_expl: Tensor [B, Freq, d]            # XAI-guided refined Key vectors (e.g. DeepLIFT)
│
├── Layers (repeated L times):
│   ├── XAI-Guided Self-Attention:
│   │   ├── AttnScore = Qexpl · Kexplᵀ / √d
│   │   ├── AttnWeights = softmax(AttnScore)
│   │   └── AttnOut = AttnWeights · Value(x)
│   ├── Residual Connection
│   ├── RMSNorm
│   ├── Feed Forward Network (GeGLU activation)
│   ├── Residual Connection
│   └── RMSNorm
│
└── Output:
    └── x_out: Tensor [B, Freq, d]             # Refined token representations
"""