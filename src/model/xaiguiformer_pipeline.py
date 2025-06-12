"""
src/model/xaiguiformer_pipeline.py - PIPELINE PRINCIPAL avec explainer corrigé - NOMS ORIGINAUX
"""

import torch
import torch.nn as nn

from model.transformer_used_two_times import SharedTransformerEncoder
from model.explainers import MultiLayerDeepLiftExplainer
from model.losses import XAIGuidedLoss, compute_class_weights
from model.gnn import ConnectomeEncoder


class XaiGuiFormer(nn.Module):
    def __init__(self, config, training_graphs=None):
        super().__init__()
        self.config = config

        # Dimensions du modèle
        self.embedding_dim = config.model.dim_node_feat
        self.num_heads = config.model.num_head
        self.num_layers = config.model.num_transformer_layer
        self.dropout = config.model.dropout
        self.num_classes = config.model.num_classes

        # ConnectomeEncoder (GNN)
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

        # ✅ Transformer compatible Captum avec nom original
        self.shared_transformer = SharedTransformerEncoder(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # Loss avec pondération des classes
        if training_graphs is not None:
            weights = compute_class_weights(training_graphs)
        else:
            weights = None
        self.alpha = config.train.criterion.alpha
        self.loss_fn = XAIGuidedLoss(alpha=self.alpha, class_weights=weights)

        # Têtes de classification
        self.classifier_coarse = nn.Linear(self.embedding_dim, self.num_classes)
        self.classifier_refined = nn.Linear(self.embedding_dim, self.num_classes)

        # ✅ Explainer compatible Captum avec nom original
        self.explainer = None  # Initialisé après premier forward

    def _init_explainer_if_needed(self):
        """Initialise l'explainer compatible Captum"""
        if self.explainer is None:
            self.explainer = MultiLayerDeepLiftExplainer(self)
            print("✅ MultiLayerDeepLiftExplainer initialized")

    def _combine_demographics(self, age, gender):
        """Combine age et gender en demographic_info"""
        return torch.cat([age, gender], dim=1)  # [B, 2]

    def forward(self, data, freq_bounds, age, gender, y_true=None):
        """
        ✅ FORWARD CORRIGÉ avec explainer Captum compatible
        """
        B = age.shape[0] if age.dim() > 0 else 1
        demographic_info = self._combine_demographics(age, gender)
        
        # Step 1: ConnectomeEncoder 
        x_embeddings = self.connectome_encoder(data)  # [B, Freq, d]
        
        # Step 2: Premier passage - Standard
        x_coarse = self.shared_transformer(
            x_embeddings, 
            freq_bounds, 
            demographic_info, 
            explanations=None,
            mode='standard'
        )
        logits_coarse = self.classifier_coarse(x_coarse.mean(dim=1))
        
        # Step 3: Explainer ✅ NOUVEAU FORMAT
        explanations = None
        if y_true is not None:
            self._init_explainer_if_needed()
            try:
                explanations = self.explainer.get_explanations(
                    x_embeddings, y_true, freq_bounds, age, gender
                )
                print(f"✅ Generated {len(explanations)} layer explanations")
            except Exception as e:
                print(f"Warning: Explainer failed: {e}")
                explanations = None
        
        # Step 4: Deuxième passage - XAI-guided
        if explanations is not None:
            # ✅ NOUVEAU : Utiliser explanations directement dans le transformer
            x_refined = self.shared_transformer(
                x_embeddings,
                freq_bounds=None,
                demographic_info=None,
                explanations=explanations,
                mode='xai_guided'
            )
        else:
            # Mode fallback si pas d'explanations
            x_refined = self.shared_transformer(
                x_embeddings,
                freq_bounds=None,
                demographic_info=None,
                explanations=None,
                mode='xai_guided'
            )
            
        logits_refined = self.classifier_refined(x_refined.mean(dim=1))
        
        if y_true is not None:
            return self.loss_fn(logits_coarse, logits_refined, y_true)
        else:
            return logits_coarse, logits_refined