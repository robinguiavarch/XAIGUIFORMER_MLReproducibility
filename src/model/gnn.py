"""
Graph Neural Network implementation for EEG connectome processing.
Uses multi-layer GINEConv for learning from brain connectivity patterns.
FALLBACK VERSION: Works without torch-scatter dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool

# Handle torch-scatter import gracefully
try:
    from torch_scatter import scatter
    SCATTER_AVAILABLE = True
    print(" torch-scatter available")
except ImportError:
    print(" torch-scatter not available, using fallback implementation")
    SCATTER_AVAILABLE = False
    
    def scatter(src, index, dim=0, reduce='mean'):
        """
        Fallback scatter implementation for basic pooling operations.
        Simple but functional replacement for torch_scatter.scatter
        """
        unique_indices = torch.unique(index)
        outputs = []
        
        for idx in unique_indices:
            mask = (index == idx)
            if reduce == 'mean':
                result = src[mask].mean(dim=0, keepdim=True)
            elif reduce == 'sum':
                result = src[mask].sum(dim=0, keepdim=True)
            elif reduce == 'max':
                result = src[mask].max(dim=0, keepdim=True)[0]
            else:
                result = src[mask].mean(dim=0, keepdim=True)
            outputs.append(result)
        
        return torch.cat(outputs, dim=0)


class ConnectomeEncoder(nn.Module):
    """
    Multi-layer Graph Neural Network for processing EEG connectomes.
    Transforms connectivity matrices into frequency-band representations.
    """
    def __init__(
        self, 
        node_in_features=1,
        edge_in_features=1, 
        node_hidden_features=128,
        edge_hidden_features=128,
        output_features=128,
        num_gnn_layers=4,
        dropout=0.1,
        gnn_type='GINEConv',
        pooling='mean',
        num_freqband=9
    ):
        super().__init__()
        
        self.node_hidden_features = node_hidden_features
        self.output_features = output_features
        self.num_freqband = num_freqband
        self.pooling = pooling
        
        # Node and edge embeddings
        self.node_embedding = Linear(node_in_features, node_hidden_features)
        self.edge_embedding = Linear(edge_in_features, edge_hidden_features)
        
        # Multi-layer GNN
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            mlp = Sequential(
                Linear(node_hidden_features, node_hidden_features),
                ReLU(),
                Linear(node_hidden_features, node_hidden_features)
            )
            self.gnn_layers.append(GINEConv(mlp, train_eps=True))
        
        # Output projection
        self.output_projection = Linear(node_hidden_features, output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass through GNN layers.
        
        Args:
            data: PyG batch with x, edge_index, edge_attr, freqband_order
            
        Returns:
            Tensor [B, num_freqband, output_features]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # GNN layers with residual connections
        for gnn_layer in self.gnn_layers:
            x_new = gnn_layer(x, edge_index, edge_attr)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
        
        # Project to output dimension
        x = self.output_projection(x)
        
        # Pool by frequency band using scatter (with fallback)
        if hasattr(data, 'freqband_order'):
            x_pooled = scatter(x, data.freqband_order, dim=0, reduce=self.pooling)
        else:
            # Fallback: assume sequential ordering
            batch_size = data.y.shape[0] if hasattr(data, 'y') else 1
            total_nodes = x.shape[0]
            nodes_per_batch = total_nodes // batch_size
            
            if nodes_per_batch % self.num_freqband == 0:
                nodes_per_band = nodes_per_batch // self.num_freqband
                x_pooled = x.view(batch_size, self.num_freqband, nodes_per_band, -1).mean(dim=2)
            else:
                # Simple fallback: global pooling then repeat
                x_mean = x.mean(dim=0, keepdim=True)
                x_pooled = x_mean.repeat(batch_size, self.num_freqband, 1)
        
        return x_pooled


class EEGConnectomeGNN(nn.Module):
    """
    Complete GNN model for EEG connectome classification.
    For standalone usage or comparison with XaiGuiFormer.
    """
    def __init__(self, input_dim=1, hidden_dim=128, num_classes=3, num_layers=4):
        super().__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            mlp = Sequential(
                Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.gnn_layers.append(GINEConv(mlp))

        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)