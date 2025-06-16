"""
losses.py

This file contains:
- Class weight computation for imbalanced datasets.
- A weighted version of CrossEntropyLoss.
- The XAIGuidedLoss class, which combines coarse and refined outputs
  according to the XAIguiFormer paper (without attention regularization for now).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


def compute_class_weights(graphs):
    """
    Compute inverse-frequency class weights from a list of PyTorch Geometric graphs.

    Args:
        graphs (List[torch_geometric.data.Data]): List of graphs with 'y' attribute as class label.

    Returns:
        torch.Tensor: Class weights of shape [num_classes], suitable for CrossEntropyLoss.
    """
    labels = [data.y.item() for data in graphs]
    counter = Counter(labels)
    total = sum(counter.values())
    num_classes = max(counter.keys()) + 1

    weights = [total / counter.get(cls, 1) for cls in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def weighted_cross_entropy_loss(outputs, targets, class_weights):
    """
    Compute a weighted CrossEntropyLoss using provided class weights.

    Args:
        outputs (torch.Tensor): Model logits of shape [batch_size, num_classes].
        targets (torch.Tensor): Ground truth labels of shape [batch_size].
        class_weights (torch.Tensor): Class weights of shape [num_classes].

    Returns:
        torch.Tensor: Scalar loss averaged over the batch.
    """
    return F.cross_entropy(outputs, targets, weight=class_weights)


class XAIGuidedLoss(nn.Module):
    """
    Combines two classification outputs (coarse and refined) using a weighted sum.

    This loss is designed for models like XaiGuiFormer that output both
    intermediate ("coarse") and final ("refined") predictions.

    Args:
        class_weights (torch.Tensor): Class weights to address class imbalance.
        alpha (float): Weighting factor between coarse and refined losses.
                       alpha=1.0 uses only refined loss; alpha=0.0 uses only coarse.
    """

    def __init__(self, class_weights, alpha=0.5):
        super(XAIGuidedLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha

    def forward(self, logits_coarse, logits_refined, targets):
        """
        Compute the weighted combined loss from coarse and refined outputs.

        Args:
            logits_coarse (torch.Tensor): Coarse logits of shape [B, C].
            logits_refined (torch.Tensor): Refined logits of shape [B, C].
            targets (torch.Tensor): Ground truth class labels of shape [B].

        Returns:
            torch.Tensor: Scalar total loss value.
        """
        loss_coarse = weighted_cross_entropy_loss(logits_coarse, targets, self.class_weights)
        loss_refined = weighted_cross_entropy_loss(logits_refined, targets, self.class_weights)

        return (1 - self.alpha) * loss_coarse + self.alpha * loss_refined
