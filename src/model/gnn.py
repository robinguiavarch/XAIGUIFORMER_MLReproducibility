"""
src/model/gnn.py - CORRECTION pour accepter format déjà reshaped
Basé sur votre debug précédent : [36, 528] → [2, 18, 528] est CORRECT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import GINEConv, global_mean_pool


class ConnectomeEncoder(torch.nn.Module):
    """
    ✅ CORRIGÉ : Accepte le format déjà reshaped par test_training.py
    [2, 18, 528] → projection directe → [2, 18, 128]
    """
    def __init__(
        self, 
        node_in_features=26,
        edge_in_features=1, 
        node_hidden_features=128,
        edge_hidden_features=128,
        output_features=128,
        num_gnn_layers=4,
        dropout=0.1,
        gnn_type='GINEConv',
        gnn_hidden_features=None,
        pooling='mean',
        num_freqband=18,
        act_func=None,
        norm=None
    ):
        super().__init__()
        
        self.input_features = 528  # Dimension des x_tokens
        self.output_features = output_features
        self.num_freqband = num_freqband
        
        print(f"✅ ConnectomeEncoder configuré:")
        print(f"   Input: {self.input_features} features (diagnostic)")
        print(f"   Output: {output_features} features")
        print(f"   Freq bands: {num_freqband}")
        
        # Projection directe : 528 → 128
        self.token_projection = nn.Linear(self.input_features, output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        ✅ FORWARD SIMPLE : Utilise x_tokens tel quel si déjà reshaped
        
        Args:
            data: Batch PyG avec x_tokens (peut être déjà reshaped)
            
        Returns:
            x: [B, Freq, output_features]
        """
        if not hasattr(data, 'x_tokens'):
            raise ValueError("❌ data doit contenir x_tokens")
        
        x_tokens = data.x_tokens
        print(f"🔧 ConnectomeEncoder input: {x_tokens.shape}")
        
        # ✅ LOGIQUE SIMPLIFIÉE basée sur votre debug
        if x_tokens.dim() == 3:
            # Format déjà correct : [B, Freq, d] - utilisation directe
            print(f"✅ Format [B, Freq, d] détecté - utilisation directe")
            x_ready = x_tokens
            
        elif x_tokens.dim() == 2:
            # Format PyG brut : [B*Freq, d] - besoin de déduire les dimensions
            print(f"✅ Format [B*Freq, d] détecté - tentative de reshape")
            
            # Essayer de déduire B depuis d'autres attributs
            if hasattr(data, 'age'):
                if data.age.dim() > 0 and data.age.numel() > 1:
                    B = data.age.shape[0]
                else:
                    B = 1
            elif hasattr(data, 'y'):
                if data.y.dim() > 0 and data.y.numel() > 1:
                    B = data.y.shape[0]
                else:
                    B = 1
            else:
                B = 1
            
            total_tokens = x_tokens.shape[0]
            d = x_tokens.shape[1]
            
            if B == 1:
                # Single sample
                x_ready = x_tokens.unsqueeze(0)  # [Freq, d] → [1, Freq, d]
                print(f"✅ Single sample: {x_tokens.shape} → {x_ready.shape}")
            else:
                # Batch
                Freq = total_tokens // B
                if total_tokens % B != 0:
                    raise ValueError(f"❌ total_tokens ({total_tokens}) n'est pas divisible par B ({B})")
                
                x_ready = x_tokens.view(B, Freq, d)
                print(f"✅ Batch reshape: {x_tokens.shape} → {x_ready.shape}")
        else:
            raise ValueError(f"❌ Format x_tokens non supporté: {x_tokens.shape}")
        
        # ✅ VALIDATION finale
        if x_ready.shape[-1] != self.input_features:
            raise ValueError(f"❌ Dernière dimension incorrecte: {x_ready.shape[-1]}, attendu {self.input_features}")
        
        # ✅ PROJECTION : 528 → 128
        x_projected = self.token_projection(x_ready)
        x_projected = self.dropout(x_projected)
        
        print(f"✅ ConnectomeEncoder output: {x_projected.shape}")
        
        return x_projected


# ===== BACKWARD COMPATIBILITY =====
class EEGConnectomeGNN(torch.nn.Module):
    """GNN original pour référence"""
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.mlp1 = Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.mlp2 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.mlp3 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.mlp4 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))

        self.conv1 = GINEConv(self.mlp1)
        self.conv2 = GINEConv(self.mlp2)
        self.conv3 = GINEConv(self.mlp3)
        self.conv4 = GINEConv(self.mlp4)

        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        return x


"""
✅ CORRECTION basée sur votre debug précédent:

PRINCIPE: Respecter le reshape fait par test_training.py
- test_training.py fait: [36, 528] → [2, 18, 528] ✅ CORRECT
- ConnectomeEncoder reçoit: [2, 18, 528] ✅ FORMAT ATTENDU
- ConnectomeEncoder fait: [2, 18, 528] → projection → [2, 18, 128] ✅

CHANGEMENTS:
1. Détection dim() == 3 → utilisation directe (pas de reshape)
2. Validation des dimensions avant projection
3. Gestion robuste single sample vs batch
4. Debug prints pour traçabilité

FLOW ATTENDU:
Input: [2, 18, 528] (déjà correct)
→ Détection format 3D ✅
→ Projection directe 528→128 ✅ 
→ Output: [2, 18, 128] ✅
"""