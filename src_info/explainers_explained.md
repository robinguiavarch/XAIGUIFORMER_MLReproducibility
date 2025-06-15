# 🧠 Understanding `explainers.py` – DeepLIFT-based Attribution for XAIguiFormer

This document explains how the module `explainers.py` works, focusing on the **DeepLIFT explainer** used to generate token-level importance scores for the XAI-guided attention mechanism in XAIguiFormer.

---

## 📌 Scientific Background – DeepLIFT (Shrikumar et al., 2017)

DeepLIFT (Deep Learning Important FeaTures) is an **XAI technique** that attributes output predictions to input features by **comparing activations to a reference (baseline)** and computing contributions based on differences.

Given:

- a model $f(x)$,
- an input $x$,
- and a baseline $x_0$ (e.g., all-zero input),

the contribution of each feature is:

$ \text{Contribution}_i = \frac{f(x) - f(x_0)}{x_i - x_{0,i}} \cdot (x_i - x_{0,i}) $


This gives per-feature relevance scores.

In XAIguiFormer, DeepLIFT is applied to the **vanilla transformer's classifier output** to identify which parts of the token sequence (connectome embeddings) are most important for prediction.

---

## 🧩 Code Breakdown – `Explainer` Class

```python
from captum.attr import DeepLift

class Explainer:
    def __init__(self, model, classifier_head):
        self.model = model.eval()
        self.classifier = classifier_head.eval()

        def forward_fn(x):
            x_encoded = self.model(x)              # [B, F, d]
            x_pooled = x_encoded.mean(dim=1)       # mean over tokens
            logits = self.classifier(x_pooled)     # [B, C]
            return logits

        self.explainer = DeepLift(forward_fn)
```

- Wraps the **Vanilla Transformer + Coarse Classifier** as a function `forward_fn`.
- Applies mean pooling before classification.
- Captum’s `DeepLift` is applied on this forward function.

---

```python
    def compute_explanations(self, x_input, target_labels):
        baseline = torch.zeros_like(x_input)  # all-zero input
        attributions = self.explainer.attribute(inputs=x_input,
                                                baselines=baseline,
                                                target=target_labels)
        return attributions
```

- Computes the attributions (importance scores) of each token in `x_input`.
- Outputs a tensor of the same shape: `attributions ∈ ℝ^{B × F × d}`.

---

## 🖼️ Module Architecture

```
Explainer/
│
├── Input:
│   ├── x_input: Tensor [B, Freq, d]       # e.g., q_rot or k_rot (before XAI attention)
│   └── target_labels: Tensor [B]          # ground-truth labels
│
├── Process:
│   ├── Wrap forward pass of:
│   │     → VanillaTransformerEncoder
│   │     → MeanPooling over tokens
│   │     → Classifier head
│   ├── Instantiate Captum DeepLift
│   └── Compute attributions w.r.t. baseline (zero input)
│
└── Output:
    └── attributions: Tensor [B, Freq, d]  # token-level feature importance
```

---

## ✅ Summary

- DeepLIFT identifies which **EEG frequency band tokens** (and which embedding dimensions) contribute the most to the vanilla transformer prediction.
- This importance map is injected into the **XAI-guided attention** as refined $Q$ and $K$.
- Enables explainable and focused attention, improving interpretability and performance.

---

This module operationalizes the key idea behind **Section 4.4 – XAI Guided Self-Attention** in the XAIguiFormer paper, and connects the vanilla encoder to its explainable counterpart.
