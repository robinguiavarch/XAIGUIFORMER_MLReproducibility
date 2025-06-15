"""
losses.py

Ce fichier contient :
- Le calcul des poids de classe pour données déséquilibrées.
- Une version pondérée de la CrossEntropyLoss.
- Une classe XAIGuidedLoss combinant les sorties coarse et refined
  conformément à l'article XAIguiFormer (sans régularisation attention pour l’instant).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


def compute_class_weights(graphs):
    """
    Calcule les poids inverses de fréquence pour chaque classe à partir
    d'une liste de graphes torch_geometric contenant un attribut 'y'.

    Args:
        graphs (list of torch_geometric.data.Data): Liste des graphes avec leurs labels.

    Returns:
        torch.Tensor: Poids des classes (de taille [num_classes]), à utiliser avec CrossEntropyLoss.
    """
    labels = [data.y.item() for data in graphs]
    counter = Counter(labels)
    total = sum(counter.values())
    num_classes = max(counter.keys()) + 1

    weights = [total / counter.get(cls, 1) for cls in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def weighted_cross_entropy_loss(outputs, targets, class_weights):
    """
    Calcule la perte CrossEntropy pondérée en fonction des poids de classes fournies.

    Args:
        outputs (torch.Tensor): Logits du modèle (batch_size, num_classes).
        targets (torch.Tensor): Labels cibles (batch_size).
        class_weights (torch.Tensor): Poids des classes (num_classes).

    Returns:
        torch.Tensor: Perte moyenne du batch.
    """
    return F.cross_entropy(outputs, targets, weight=class_weights)


class XAIGuidedLoss(nn.Module):
    """
    Combine deux sorties de classification (coarse et refined) avec pondération alpha.

    Args:
        class_weights (Tensor): Poids des classes (pour déséquilibre).
        alpha (float): Pondération entre coarse et refined loss.
    """
    def __init__(self, class_weights, alpha=0.5):
        super(XAIGuidedLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = alpha

    def forward(self, logits_coarse, logits_refined, targets):
        """
        Args:
            logits_coarse (Tensor): Logits coarse [B, C]
            logits_refined (Tensor): Logits refined [B, C]
            targets (Tensor): Labels [B]

        Returns:
            Tensor: Perte totale.
        """
        loss_coarse = weighted_cross_entropy_loss(logits_coarse, targets, self.class_weights)
        loss_refined = weighted_cross_entropy_loss(logits_refined, targets, self.class_weights)

        return (1 - self.alpha) * loss_coarse + self.alpha * loss_refined
