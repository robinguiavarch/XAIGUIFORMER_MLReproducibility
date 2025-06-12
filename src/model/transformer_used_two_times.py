import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTransformerLayer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1, use_drofe=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_drofe = use_drofe  # Ajout du flag

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.ff_linear1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.ff_linear2 = nn.Linear(embedding_dim, embedding_dim)

        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        from model.dRoFE_embedding import dRoFEEmbedding
        self.drofe = dRoFEEmbedding(embedding_dim)

    def geglu(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1) * x2

    def forward(self, x, context_info=None):
        B, Freq, d = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_drofe and context_info is not None:
            freq_bounds = context_info[:, :Freq, :2]
            demographics = context_info[:, Freq:Freq+2, :1].squeeze(-1)

            freq_bounds_single = freq_bounds[0]
            age = demographics[:, 0:1]
            gender = demographics[:, 1:2]

            q_rot = self.drofe(q, freq_bounds_single, age, gender)
            k_rot = self.drofe(k, freq_bounds_single, age, gender)
        else:
            q_rot = q
            k_rot = k

        q_rot = q_rot.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        k_rot = k_rot.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        self.attn_weights_cache = attn_weights.detach()

        attn_weights = self.attn_dropout(attn_weights)

        self.attn_weights_cache = attn_weights.clone().detach()

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Freq, d)
        attn_output = self.out_proj(attn_output)

        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff_out = self.ff_linear2(self.geglu(self.ff_linear1(x)))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class SharedTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=12, dropout=0.1, use_drofe=True):
        super().__init__()

        self.layers = nn.ModuleList([
            SharedTransformerLayer(embedding_dim, num_heads, dropout, use_drofe)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(embedding_dim)
        print(f"✅ SharedTransformerEncoder: {num_layers} layers - use_drofe={use_drofe}")

    def _pack_context(self, freq_bounds, demographic_info):
        B = demographic_info.shape[0]
        Freq = freq_bounds.shape[0]

        freq_bounds_batched = freq_bounds.unsqueeze(0).repeat(B, 1, 1)
        demographics_padded = demographic_info.unsqueeze(-1).repeat(1, 1, 2)

        context_info = torch.cat([freq_bounds_batched, demographics_padded], dim=1)
        return context_info

    def forward(self, x, freq_bounds=None, demographic_info=None, explanations=None, mode='standard'):
        context_info = None
        if mode == 'standard' and freq_bounds is not None and demographic_info is not None:
            context_info = self._pack_context(freq_bounds, demographic_info)

        for i, layer in enumerate(self.layers):
            if mode == 'xai_guided' and explanations is not None and i < len(explanations):
                x_input = explanations[i]
            else:
                x_input = x

            x = layer(x_input, context_info)

        return self.final_norm(x)


class CaptumExplainerWrapper(nn.Module):
    def __init__(self, transformer, classifier):
        super().__init__()
        self.transformer = transformer
        self.classifier = classifier

    def forward(self, x, context_info=None):
        mode = 'standard' if context_info is not None else 'xai_guided'

        freq_bounds = None
        demographic_info = None

        if context_info is not None:
            B, total_len, _ = context_info.shape
            Freq = total_len - 2

            freq_bounds = context_info[0, :Freq, :2]
            demographics_raw = context_info[:, Freq:Freq+2, 0]
            demographic_info = demographics_raw

        x_out = self.transformer(x, freq_bounds, demographic_info, None, mode)
        logits = self.classifier(x_out.mean(dim=1))

        return logits
