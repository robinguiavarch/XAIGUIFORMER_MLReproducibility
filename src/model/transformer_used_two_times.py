import torch
import torch.nn as nn
import torch.nn.functional as F


class XAIGuiAttention(nn.Module):
    """
    Multi-head self-attention layer with optional XAI-guided injection.

    This module allows substitution of the query and key projections
    with external explanation vectors (e.g., from DeepLIFT),
    and optionally applies dRoFE-based rotation based on frequency and demographics.
    """
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1, use_drofe=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_drofe = use_drofe

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)

        if use_drofe:
            from model.dRoFE_embedding import dRoFEEmbedding
            self.drofe = dRoFEEmbedding(embedding_dim)

    def forward(self, x, context_info=None, explanation=None):
        """
        Forward pass with optional explanation-based injection.

        Args:
            x (Tensor): Input embeddings [B, Freq, d].
            context_info (Tensor, optional): Packed frequency + demographic info [B, Freq+2, 2].
            explanation (Tensor, optional): Explanation tensor [B, Freq, d] to override Q and K.

        Returns:
            Tensor: Output embeddings [B, Freq, d].
        """
        B, Freq, d = x.shape

        # Compute standard Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Replace Q and K if explanation is provided
        if explanation is not None:
            q = self.q_proj(explanation)
            k = self.k_proj(explanation)

        # Apply dRoFE if enabled
        if self.use_drofe and context_info is not None:
            freq_bounds = context_info[:, :Freq, :2]
            demographics = context_info[:, Freq:Freq + 2, :1].squeeze(-1)
            age = demographics[:, 0:1]
            gender = demographics[:, 1:2]
            freq_bounds_single = freq_bounds[0]

            q = self.drofe(q, freq_bounds_single, age, gender)
            k = self.drofe(k, freq_bounds_single, age, gender)

        # Multi-head attention
        q = q.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Freq, d)

        return self.out_proj(attn_output)


class SharedTransformerLayer(nn.Module):
    """
    Single transformer layer supporting standard and XAI-guided modes.
    Includes optional dRoFE-aware attention and GEGLU feedforward.
    """
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1, use_drofe=True):
        super().__init__()

        self.attention = XAIGuiAttention(embedding_dim, num_heads, dropout, use_drofe)
        self.ff_linear1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.ff_linear2 = nn.Linear(embedding_dim, embedding_dim)

        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def geglu(self, x):
        """
        GEGLU activation (Gated GELU) for feed-forward networks.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1) * x2

    def forward(self, x, context_info=None, explanation=None):
        """
        Forward pass through attention and feedforward blocks.

        Args:
            x (Tensor): Input embeddings [B, Freq, d].
            context_info (Tensor, optional): Packed frequency + demographic info.
            explanation (Tensor, optional): XAI-guided explanation vector for attention.

        Returns:
            Tensor: Output after attention and FFN layers.
        """
        attn_output = self.attention(x, context_info, explanation)
        x = self.norm1(x + self.dropout(attn_output))

        ff_out = self.ff_linear2(self.geglu(self.ff_linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))

        return x


class SharedTransformerEncoder(nn.Module):
    """
    Stack of shared transformer layers supporting standard and XAI-guided modes.

    Allows integration of external explanations at each layer,
    and supports demographic conditioning via dRoFE.
    """
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=12, dropout=0.1, use_drofe=True):
        super().__init__()

        self.layers = nn.ModuleList([
            SharedTransformerLayer(embedding_dim, num_heads, dropout, use_drofe)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(embedding_dim)

    def _pack_context(self, freq_bounds, demographic_info):
        """
        Prepare and concatenate frequency bounds and demographics.

        Args:
            freq_bounds (Tensor): [Freq, 2].
            demographic_info (Tensor): [B, 2].

        Returns:
            Tensor: [B, Freq + 2, 2] packed context.
        """
        B = demographic_info.shape[0]
        Freq = freq_bounds.shape[0]

        freq_bounds_batched = freq_bounds.unsqueeze(0).repeat(B, 1, 1)
        demographics_padded = demographic_info.unsqueeze(-1).repeat(1, 1, 2)

        return torch.cat([freq_bounds_batched, demographics_padded], dim=1)

    def forward(self, x, freq_bounds=None, demographic_info=None, explanations=None, mode='standard'):
        """
        Forward pass through transformer stack.

        Args:
            x (Tensor): Input embeddings [B, Freq, d].
            freq_bounds (Tensor, optional): EEG band bounds [Freq, 2].
            demographic_info (Tensor, optional): Age and gender [B, 2].
            explanations (List[Tensor], optional): One per layer for XAI-guided mode.
            mode (str): Either 'standard' or 'xai_guided'.

        Returns:
            Tensor: Final output after all transformer layers.
        """
        context_info = None
        if mode == 'standard' and freq_bounds is not None and demographic_info is not None:
            context_info = self._pack_context(freq_bounds, demographic_info)

        for i, layer in enumerate(self.layers):
            layer_expl = explanations[i] if (mode == 'xai_guided' and explanations is not None and i < len(explanations)) else None
            x = layer(x, context_info, layer_expl)

        return self.final_norm(x)


class CaptumExplainerWrapper(nn.Module):
    """
    Wrapper to adapt transformer+classifier for Captum-based explanation generation.

    This interface matches Captum's multi-input requirement, and extracts 
    context from packed tensors during explanation attribution.
    """
    def __init__(self, transformer, classifier):
        super().__init__()
        self.transformer = transformer
        self.classifier = classifier

    def forward(self, x, context_info=None):
        """
        Forward method for Captum compatibility.

        Args:
            x (Tensor): Input embeddings [B, Freq, d].
            context_info (Tensor, optional): Packed context [B, Freq+2, 2].

        Returns:
            Tensor: Logits [B, C].
        """
        mode = 'standard' if context_info is not None else 'xai_guided'

        freq_bounds = None
        demographic_info = None

        if context_info is not None:
            B, total_len, _ = context_info.shape
            Freq = total_len - 2

            freq_bounds = context_info[0, :Freq, :2]
            demographics_raw = context_info[:, Freq:Freq + 2, 0]
            demographic_info = demographics_raw

        x_out = self.transformer(x, freq_bounds, demographic_info, explanations=None, mode=mode)
        logits = self.classifier(x_out.mean(dim=1))

        return logits
