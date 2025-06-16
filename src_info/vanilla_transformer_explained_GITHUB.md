# Understanding `vanilla_transformer.py` – Transformer Encoder Explained

This document explains the logic of `vanilla_transformer.py` line by line, in relation to the Transformer architecture used in XAIGUIFormer, focusing on the **vanilla Transformer encoder** (not XAI-guided yet).

---

## Scientific Background – Transformer Core Concepts

### Multi-Head Attention

Given an input tensor `X ∈ ℝ^{B × F × d}`, the Transformer computes:

- Queries: `Q = X @ W_q`
- Keys: `K = X @ W_k`
- Values: `V = X @ W_v`

Each attention head computes:

```
Attention(Q, K, V) = softmax((Q @ Kᵀ) / sqrt(d)) @ V
```

This captures dependencies between tokens (here: frequency bands), regardless of position.

Multiple heads are used in parallel to learn **diverse interactions**. Their outputs are concatenated and projected again.

---

### RMSNorm (Root Mean Square LayerNorm)

A lightweight alternative to LayerNorm:

```
RMSNorm(x) = x / RMS(x) * gamma
where RMS(x) = sqrt(mean(x_i^2 for i in 1..d))
```

Used instead of LayerNorm to reduce compute while preserving stability.

---

### Feedforward Network (FFN)

Each encoder block includes a 2-layer MLP applied to each token independently:

```
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
```

This increases the expressivity of the model.

---

## Code Breakdown – `VanillaTransformerEncoder`

```python
class VanillaTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        ...
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(embedding_dim)
```

- Defines a stack of vanilla Transformer encoder layers.
- Each layer includes Multi-Head Attention + Feedforward block + residual connections + RMSNorm (or LayerNorm depending on implementation).

---

```python
def forward(self, x):
    x = self.encoder(x)
    return self.layernorm(x)
```

- Input: `x ∈ ℝ^{B × F × d}` — sequence of connectome tokens
- Output: same shape, updated token representations

---

## Architecture Diagram

```
Input → Connectome Tokenizer → [Vanilla Multi-Head Attention → RMSNorm → Feed Forward → RMSNorm] → Output
```

This encoder block is stacked `L` times (e.g. 4 layers) to form a deep Transformer.

---

## Summary

- **Q, K, V** are computed directly from the input tokens.
- **Attention** learns relations between EEG frequency bands.
- **RMSNorm** stabilizes training with lower computational cost.
- **Feedforward** layers expand the representational power of each token.
- The full encoder outputs `X' ∈ ℝ^{B × F × d}`, ready for classification or refinement.

---

This module is the foundation of XAIGUIFormer, later enhanced by XAI-guided attention and demographic encoding (dRoFE).