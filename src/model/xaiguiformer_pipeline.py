import torch
import torch.nn as nn

from model.dRoFE_embedding import dRoFEEmbedding
from model.vanilla_transformer import VanillaTransformerEncoder
from model.xai_guided_transformer import XAIGuidedTransformerEncoder
from model.losses import XAIGuidedLoss, compute_class_weights
from model.gnn import EEGConnectomeGNN
from model.explainers import DeepLiftExplainer


class XaiGuiFormer(nn.Module):
    def __init__(self, config, training_graphs=None):
        super().__init__()
        self.config = config

        # === Dimensions du modèle ===
        self.embedding_dim = config.model.dim_node_feat
        self.num_heads = config.model.num_head
        self.num_layers = config.model.num_transformer_layer
        self.dropout = config.model.dropout
        self.num_classes = config.model.num_classes

        # === Modules ===
        self.gnn = EEGConnectomeGNN(
            input_dim=config.model.num_node_feat,
            hidden_dim=self.embedding_dim,
            num_classes=self.num_classes
        )

        self.drofe = dRoFEEmbedding(embedding_dim=self.embedding_dim)

        self.vanilla_transformer = VanillaTransformerEncoder(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        self.xai_transformer = XAIGuidedTransformerEncoder(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # === Loss avec pondération des classes ===
        if training_graphs is not None:
            weights = compute_class_weights(training_graphs)
        else:
            weights = None
        self.alpha = config.train.criterion.alpha
        self.loss_fn = XAIGuidedLoss(alpha=self.alpha, class_weights=weights)

        # === Projection des tokens depuis 528 → 128 ===
        self.token_projection = nn.Linear(in_features=528, out_features=self.embedding_dim)

        # === Projections pour Query et Key ===
        self.W_q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_k = nn.Linear(self.embedding_dim, self.embedding_dim)

        # === Têtes de classification ===
        self.classifier_coarse = nn.Linear(self.embedding_dim, self.num_classes)
        self.classifier_refined = nn.Linear(self.embedding_dim, self.num_classes)

        # === Explainer DeepLIFT ===
        self.explainer = DeepLiftExplainer(self)

    def forward(self, data, freq_bounds, age, gender, y_true=None):
        """
        Args:
            data        : batch de graphes (PyG) → contient x_tokens
            freq_bounds : [Freq, 2]
            age         : [B, 1]
            gender      : [B, 1]
            y_true      : [B] or None
        Returns:
            loss or logits
        """

        # 1. GNN → extraction de tokens connectome pour chaque bande f
        x_raw = data.x_tokens  # [B, Freq, 528] ou [Freq, 528] si batch size = 1
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(0)  # [Freq, 528] → [1, Freq, 528]

        x_raw = self.token_projection(x_raw)  # [B, Freq, 128]

        # 2. dRoFE sur Q/K
        q = self.W_q(x_raw)
        k = self.W_k(x_raw)

        q_rot = self.drofe(q, freq_bounds, age, gender)
        k_rot = self.drofe(k, freq_bounds, age, gender)

        # 3. Vanilla Transformer
        x_coarse = self.vanilla_transformer(q_rot)
        logits_coarse = self.classifier_coarse(x_coarse.mean(dim=1))

        # 4. Explainer DeepLIFT (utilise logits coarse pour extraire importance)
        if y_true is not None:
            q_expl = self.explainer.get_explanations(q_rot, y_true, freq_bounds, age, gender)
            k_expl = self.explainer.get_explanations(k_rot, y_true)
        else:
            q_expl = q_rot.detach()
            k_expl = k_rot.detach()

        # 5. XAI-guided Transformer
        x_refined = self.xai_transformer(x_raw, q_expl, k_expl)
        logits_refined = self.classifier_refined(x_refined.mean(dim=1))

        # 6. Sortie
        if y_true is not None:
            return self.loss_fn(logits_coarse, logits_refined, y_true)
        else:
            return logits_coarse, logits_refined


# Architecture du modèle
"""
XaiGuiFormer/
│
├── Inputs:
│   ├── data         : Batch de graphes PyG (1 par bande EEG)
│   ├── freq_bounds  : Tensor [Freq, 2]         # EEG frequency band bounds: [f_l, f_u]
│   ├── age          : Tensor [B, 1]            # Patient ages (normalized)
│   ├── gender       : Tensor [B, 1]            # Patient genders (binary or scalar)
│   └── y_true       : Tensor [B] or None       # Ground truth labels
│
├── Step 1 – Tokenization par GNN (Connectome Tokenizer):
│   ├── Pour chaque sujet et chaque bande EEG f:
│   │     └── graphe G_f → EEGConnectomeGNN → embedding x_f ∈ ℝᵈ
│   └── Agrégation → x_raw ∈ ℝ^{B × Freq × d}
│
├── Step 2 – dRoFE Embedding:
│   ├── Projections Q = W_q(x_raw), K = W_k(x_raw)
│   ├── q_rot = dRoFE(Q, freq_bounds, age, gender)
│   └── k_rot = dRoFE(K, freq_bounds, age, gender)
│
├── Step 3 – Vanilla Transformer Encoder:
│   ├── x_coarse = VanillaTransformer(q_rot)
│   └── logits_coarse = classifier_coarse(mean(x_coarse, dim=1))   # [B, C]
│
├── Step 4 – DeepLIFT Explainer:
│   ├── q_expl = DeepLift(q_rot, target=y_true)                     # [B, Freq, d]
│   └── k_expl = DeepLift(k_rot, target=y_true)
│
├── Step 5 – XAI-Guided Transformer:
│   ├── x_refined = XAIGuidedTransformer(x_raw, q_expl, k_expl)
│   └── logits_refined = classifier_refined(mean(x_refined, dim=1)) # [B, C]
│
├── Step 6 – Loss Computation (optional):
│   └── total_loss = (1 - α) * CE(logits_coarse, y_true) + α * CE(logits_refined, y_true)
│
└── Outputs:
    ├── If y_true is None:
    │     └── logits_coarse: Tensor [B, C]
    │     └── logits_refined: Tensor [B, C]
    └── Else:
          └── total_loss: scalar
"""