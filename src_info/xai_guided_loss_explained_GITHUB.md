# Understanding `losses.py` – XAI-Guided Loss Explained

This document explains the role and structure of the `XAIGuidedLoss` class used in XAIguiFormer. It defines a **joint loss function** combining the predictions from both the **vanilla transformer** and the **XAI-guided transformer**, encouraging complementary learning.

---

## Theoretical Background

XAIGuiFormer uses **two prediction heads**:
- A **coarse prediction** from the vanilla transformer (standard attention)
- A **refined prediction** from the XAI-guided transformer (attention guided by DeepLIFT/IG)

To train both simultaneously, the model combines both losses:

```
L = (1 - alpha) * CE(y_coarse, y_true) + alpha * CE(y_refined, y_true)
```

Where:
- `y_coarse` = logits from vanilla transformer
- `y_refined` = logits from XAI-guided transformer
- `y_true` = ground truth label
- `alpha ∈ [0,1]` = trade-off parameter

---

## Code Breakdown – `XAIGuidedLoss`

```python
class XAIGuidedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, y_coarse, y_refined, y_true):
        loss_coarse = self.ce(y_coarse, y_true)
        loss_refined = self.ce(y_refined, y_true)
        total_loss = (1 - self.alpha) * loss_coarse + self.alpha * loss_refined
        return total_loss
```

- `y_coarse` : logits from vanilla transformer [B, C]
- `y_refined` : logits from XAI-guided transformer [B, C]
- `y_true` : true labels (class indices) [B]
- Output: scalar total loss

---

## Architecture Diagram

```
XAIGuidedLoss/
│
├── Inputs:
│   ├── y_coarse  : Tensor [B, C]         # Logits from Vanilla Transformer
│   ├── y_refined : Tensor [B, C]         # Logits from XAI-Guided Transformer
│   └── y_true    : Tensor [B]            # Ground-truth labels
│
├── Parameters:
│   └── alpha ∈ [0, 1]                    # Trade-off parameter
│
├── Computation:
│   ├── L_coarse = CE(y_coarse, y_true)
│   ├── L_refined = CE(y_refined, y_true)
│   └── L_total = (1 - alpha) * L_coarse + alpha * L_refined
│
└── Output:
    └── total_loss: scalar                # Final supervised loss
```

---

## Summary

- Allows the model to balance learning between the vanilla and XAI-guided paths.
- Encourages the refined attention mechanism to improve beyond the vanilla baseline.
- Easy to adjust using `alpha` (typically set between 0.3 and 0.7).

---

This joint loss is central to aligning performance with interpretability in XAIGuiFormer.