import torch.nn as nn

class XAIGuidedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: weighting parameter between coarse and refined loss
        """
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, y_coarse, y_refined, y_true):
        """
        Args:
            y_coarse: logits from vanilla transformer (B, C)
            y_refined: logits from XAI-guided transformer (B, C)
            y_true: true labels (B,)
        Returns:
            total_loss: scalar
        """
        loss_coarse = self.ce(y_coarse, y_true)
        loss_refined = self.ce(y_refined, y_true)

        total_loss = (1 - self.alpha) * loss_coarse + self.alpha * loss_refined
        return total_loss


"""
XAIGuidedLoss/
│
├── Inputs:
│   ├── y_coarse  : Tensor [B, C]         # Logits from Vanilla Transformer (coarse prediction)
│   ├── y_refined : Tensor [B, C]         # Logits from XAI-Guided Transformer (refined prediction)
│   └── y_true    : Tensor [B]            # Ground-truth labels (class index)
│
├── Parameters:
│   └── alpha ∈ [0, 1]                    # Trade-off between coarse and refined supervision
│
├── Step-by-step:
│   ├── Compute:
│   │     - L_coarse = CrossEntropy(y_coarse, y_true)
│   │     - L_refined = CrossEntropy(y_refined, y_true)
│   ├── Weighted combination:
│   │     - L_total = (1 - alpha) * L_coarse + alpha * L_refined
│
└── Output:
    └── total_loss: scalar                # Joint supervised loss
"""