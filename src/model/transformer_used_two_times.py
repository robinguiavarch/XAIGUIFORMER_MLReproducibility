import torch
import torch.nn as nn
import torch.nn.functional as F


class XAIGuiAttention(nn.Module):
    """
    XAI-guided Multi-Head Attention with concurrent explanation injection.
    Explanations directly replace Query and Key projections during attention computation.
    """
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1, use_drofe=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_drofe = use_drofe

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)

        if use_drofe:
            from model.dRoFE_embedding import dRoFEEmbedding
            self.drofe = dRoFEEmbedding(embedding_dim)

    def forward(self, x, context_info=None, explanation=None):
        """
        Forward pass with optional XAI explanation injection.
        
        Args:
            x: Input tensor [B, Freq, d]
            context_info: Packed frequency bounds and demographics
            explanation: XAI explanation tensor [B, Freq, d] (if provided)
        """
        B, Freq, d = x.shape

        # Standard Q, K, V projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # XAI Injection: Replace Q and K with explanations if available
        if explanation is not None:
            q_explanation = self.q_proj(explanation)
            k_explanation = self.k_proj(explanation)
            q = q_explanation
            k = k_explanation

        # Apply dRoFE rotation if enabled and context available
        if self.use_drofe and context_info is not None:
            freq_bounds = context_info[:, :Freq, :2]
            demographics = context_info[:, Freq:Freq+2, :1].squeeze(-1)

            freq_bounds_single = freq_bounds[0]
            age = demographics[:, 0:1]
            gender = demographics[:, 1:2]

            q = self.drofe(q, freq_bounds_single, age, gender)
            k = self.drofe(k, freq_bounds_single, age, gender)

        # Multi-head attention computation
        q = q.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Freq, d)
        attn_output = self.out_proj(attn_output)

        return attn_output


class SharedTransformerLayer(nn.Module):
    """
    Transformer layer supporting both standard and XAI-guided modes.
    """
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1, use_drofe=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.attention = XAIGuiAttention(embedding_dim, num_heads, dropout, use_drofe)

        self.ff_linear1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.ff_linear2 = nn.Linear(embedding_dim, embedding_dim)

        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def geglu(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1) * x2

    def forward(self, x, context_info=None, explanation=None):
        # Self-attention with optional XAI guidance
        attn_output = self.attention(x, context_info, explanation)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward network
        ff_out = self.ff_linear2(self.geglu(self.ff_linear1(x)))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class SharedTransformerEncoder(nn.Module):
    """
    Shared transformer encoder supporting dual-pass XAI-guided processing.
    """
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=12, dropout=0.1, use_drofe=True):
        super().__init__()

        self.layers = nn.ModuleList([
            SharedTransformerLayer(embedding_dim, num_heads, dropout, use_drofe)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(embedding_dim)

    def _pack_context(self, freq_bounds, demographic_info):
        """Pack frequency bounds and demographics for dRoFE."""
        B = demographic_info.shape[0]
        Freq = freq_bounds.shape[0]

        freq_bounds_batched = freq_bounds.unsqueeze(0).repeat(B, 1, 1)
        demographics_padded = demographic_info.unsqueeze(-1).repeat(1, 1, 2)

        context_info = torch.cat([freq_bounds_batched, demographics_padded], dim=1)
        return context_info

    def forward(self, x, freq_bounds=None, demographic_info=None, explanations=None, mode='standard'):
        """
        Forward pass through transformer layers.
        
        Args:
            x: Input embeddings [B, Freq, d]
            freq_bounds: Frequency band bounds [Freq, 2]
            demographic_info: Age and gender [B, 2]
            explanations: List of layer explanations (for XAI-guided mode)
            mode: 'standard' or 'xai_guided'
        """
        context_info = None
        if mode == 'standard' and freq_bounds is not None and demographic_info is not None:
            context_info = self._pack_context(freq_bounds, demographic_info)

        for i, layer in enumerate(self.layers):
            layer_explanation = None
            if mode == 'xai_guided' and explanations is not None and i < len(explanations):
                layer_explanation = explanations[i]

            x = layer(x, context_info, layer_explanation)

        return self.final_norm(x)


class CaptumExplainerWrapper(nn.Module):
    """
    Wrapper for Captum compatibility in explanation generation.
    """
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