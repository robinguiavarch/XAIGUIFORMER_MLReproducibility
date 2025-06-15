import torch
import torch.nn as nn

from model.transformer_used_two_times import SharedTransformerEncoder
from model.explainers import MultiLayerDeepLiftExplainer  
from model.losses import XAIGuidedLoss, compute_class_weights
from model.gnn import ConnectomeEncoder


class XaiGuiFormer(nn.Module):
    """
    XAI-guided Transformer with concurrent explanation processing.
    Implements dual-pass architecture with real-time XAI guidance.
    """
    def __init__(self, config, training_graphs=None):
        super().__init__()
        self.config = config

        self.embedding_dim = config.model.dim_node_feat
        self.num_heads = config.model.num_head
        self.num_layers = config.model.num_transformer_layer
        self.dropout = config.model.dropout
        self.num_classes = config.model.num_classes

        # Ablation study flags
        self.use_xai_guidance = config.model.use_xai_guidance
        self.use_drofe = config.model.use_drofe
        self.use_demographics = config.model.use_demographics

        # Connectome encoder (GNN)
        self.connectome_encoder = ConnectomeEncoder(
            node_in_features=config.model.num_node_feat,
            edge_in_features=config.model.num_edge_feat,
            node_hidden_features=self.embedding_dim,
            edge_hidden_features=config.model.dim_edge_feat,
            output_features=self.embedding_dim,
            num_gnn_layers=config.model.num_gnn_layer,
            dropout=self.dropout,
            gnn_type=config.model.gnn_type,
            num_freqband=9
        )

        # Shared transformer encoder
        self.shared_transformer = SharedTransformerEncoder(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_drofe=self.use_drofe
        )

        # Classification heads
        self.classifier_coarse = nn.Linear(self.embedding_dim, self.num_classes)
        self.classifier_refined = nn.Linear(self.embedding_dim, self.num_classes)

        # Loss function with class weights
        if training_graphs is not None:
            weights = compute_class_weights(training_graphs)
        else:
            weights = None

        self.alpha = config.train.criterion.alpha
        self.loss_fn = XAIGuidedLoss(alpha=self.alpha, class_weights=weights)

        # XAI explainer (initialized lazily)
        self.explainer = None

    def _init_explainer_if_needed(self):
        """Initialize explainer on first use."""
        if self.explainer is None and self.use_xai_guidance:
            self.explainer = MultiLayerDeepLiftExplainer(self)  

    def _combine_demographics(self, age, gender):
        """Combine age and gender into demographic info."""
        return torch.cat([age, gender], dim=1)

    def forward(self, data, freq_bounds, age, gender, y_true=None):
        """
        Concurrent XAI-guided forward pass.
        
        Args:
            data: PyG batch data
            freq_bounds: Frequency band bounds [Freq, 2]
            age: Age values [B, 1]
            gender: Gender values [B, 1]
            y_true: True labels [B] (for training)
            
        Returns:
            If training: Combined loss value
            If inference: Tuple of (coarse_logits, refined_logits)
        """
        B = age.shape[0] if age.dim() > 0 else 1

        # Prepare inputs based on ablation flags
        demographic_info = (
            self._combine_demographics(age, gender) if self.use_demographics else None
        )
        freq_bounds_in = freq_bounds if self.use_drofe else None

        # Step 1: GNN encoding
        x_embeddings = self.connectome_encoder(data)

        # Step 2: First pass (Standard transformer)
        x_coarse = self.shared_transformer(
            x_embeddings,
            freq_bounds=freq_bounds_in,
            demographic_info=demographic_info,
            explanations=None,
            mode='standard'
        )
        logits_coarse = self.classifier_coarse(x_coarse.mean(dim=1))

        # Step 3: Concurrent XAI explanation generation
        explanations = None
        if self.use_xai_guidance and y_true is not None:
            self._init_explainer_if_needed()
            
            if self.explainer is not None:
                try:
                    explanations = self.explainer.get_explanations(
                        x_embeddings, y_true,
                        freq_bounds if self.use_drofe else freq_bounds,
                        age if self.use_demographics else torch.zeros_like(age),
                        gender if self.use_demographics else torch.zeros_like(gender)
                    )
                except Exception:
                    explanations = None

        # Step 4: Second pass (XAI-guided transformer)
        x_refined = self.shared_transformer(
            x_embeddings,
            freq_bounds=None,  # No dRoFE in second pass
            demographic_info=None,  # No demographics in second pass
            explanations=explanations if self.use_xai_guidance else None,
            mode='xai_guided'
        )
        logits_refined = self.classifier_refined(x_refined.mean(dim=1))

        # Return loss for training, logits for inference
        if y_true is not None:
            return self.loss_fn(logits_coarse, logits_refined, y_true)
        else:
            return logits_coarse, logits_refined