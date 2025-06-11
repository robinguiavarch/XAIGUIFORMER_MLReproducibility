"""
Définition d'un modèle GNN basé sur GINEConv pour apprendre des représentations
à partir de graphes de connectomes EEG (calculés à partir de matrices de connectivité).

Chaque graphe encode un sujet (ou une session EEG) avec ses connexions entre électrodes,
et le modèle prédit un label associé (ex. état clinique, tâche cognitive...).
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool

class EEGConnectomeGNN(torch.nn.Module):
    """
    Modèle de GNN basé sur deux couches GINEConv suivies d’un pooling global moyen
    et d’un classifieur linéaire.

    Args:
        input_dim (int): Dimension des features des nœuds.
        hidden_dim (int): Dimension cachée du GNN.
        num_classes (int): Nombre de classes cibles pour la classification.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EEGConnectomeGNN, self).__init__()

        # MLPs pour GINEConv
        self.mlp1 = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.mlp2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINEConv(self.mlp1)
        self.conv2 = GINEConv(self.mlp2)

        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, data):
        """
        Passe avant (forward pass) du modèle GNN.

        Args:
            data (torch_geometric.data.Data): Mini-batch de graphes.

        Returns:
            torch.Tensor: Logits de classification (batch_size x num_classes).
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # Pooling global des nœuds -> graphe
        x = self.classifier(x)
        return x
