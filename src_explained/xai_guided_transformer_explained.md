# 🧠 Understanding `xai_guided_transformer.py` – XAI-Guided Transformer Explained

This document explains the structure and purpose of the `xai_guided_transformer.py` module, which implements the **XAI-guided Transformer block** in XAIGUIFormer.

---

## 📌 Scientific Background – XAI-Guided Attention

Unlike the vanilla Transformer that uses learned Q and K projections, this module **replaces Q and K with XAI-explained importance vectors** (e.g. DeepLIFT explanations).

### 🔹 Custom Attention Mechanism

The attention score is recalculated manually using:

\[
\text{Attention}(Q_{\text{expl}}, K_{\text{expl}}, V) = \text{softmax}\left(\frac{Q_{\text{expl}} K_{\text{expl}}^T}{\sqrt{d}}\right)V
\]

Where:
- \( Q_{\text{expl}}, K_{\text{expl}} \) are the XAI-guided feature importance maps
- \( V \) is the original input tensor

This allows attention to **focus on semantically meaningful relationships** identified by an explanation method (e.g., DeepLIFT).

---

### 🔹 Feedforward Network – GeGLU

This module uses **GeGLU (Gated GELU)** activation in the feedforward network:

\[
\text{GeGLU}(x) = \text{GELU}(x_1) \cdot x_2
\]

The feedforward block is then:

\[
\text{FF}(x) = W_2 (\text{GeGLU}(W_1 x))
\]

Where \( W_1 \) splits into two heads: \( x_1, x_2 \in \mathbb{R}^{d} \).

---

### 🔹 Normalization – RMSNorm

As in the vanilla encoder, each sub-block (attention, FFN) is followed by **residual + RMSNorm** for training stability:

\[
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
\]

---

## 🧩 Code Breakdown – `XAIGuidedTransformerEncoder`

```python
class XAIGuidedTransformerLayer(nn.Module):
    def forward(self, x, q_expl, k_expl):
        # XAI-guided attention
        attn_out = softmax(q_expl @ k_expl.T / sqrt(d)) @ x
        x = RMSNorm(x + dropout(attn_out))
        
        # Feedforward with GeGLU
        ff = Linear2(GeGLU(Linear1(x)))
        x = RMSNorm(x + dropout(ff))
        return x
```

```python
class XAIGuidedTransformerEncoder(nn.Module):
    def forward(self, x, q_expl, k_expl):
        for layer in self.layers:
            x = layer(x, q_expl, k_expl)
        return RMSNorm(x)
```

---

## 🖼️ Architecture Diagram

```
Input from Vanilla Transformer → XAI (DeepLIFT) → [XAI-guided Multi-Head Attention → RMSNorm → Feed Forward (GeGLU) → RMSNorm] × L → Output
```

---

## ✅ Summary

- Uses **XAI explanations as inputs to attention**, improving interpretability.
- Employs **custom attention computation** (manual Qexpl, Kexpl dot-product).
- Incorporates **GeGLU** for better non-linear expressiveness.
- Stabilized by **RMSNorm** and residual connections.
- Forms the second-stage Transformer in XAIGuiFormer after the vanilla encoder.

---

This module enables XAIGuiFormer to integrate feature-level importance directly into the attention mechanism, improving both performance and interpretability.