"""
losses.py

Ce fichier contient les fonctions pour gérer les pertes pondérées
dans le cadre de l'entraînement sur des données déséquilibrées.
"""

import torch
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
    Calcule la perte CrossEntropy pondérée en fonction des poids de classes fournis.

    Args:
        outputs (torch.Tensor): Prédictions du modèle (logits), de taille [batch_size, num_classes].
        targets (torch.Tensor): Labels cibles (entiers), de taille [batch_size].
        class_weights (torch.Tensor): Poids des classes, de taille [num_classes].

    Returns:
        torch.Tensor: Perte moyenne sur le batch.
    """
    probs = F.log_softmax(outputs, dim=1)  # log_softmax pour stabiliser le calcul du log
    target_log_probs = probs[torch.arange(outputs.size(0)), targets]
    weights = class_weights[targets]

    loss = - weights * target_log_probs

    return loss.mean()
