import torch
import torch.nn as nn

from model.dRoFE_embedding import dRoFEEmbedding
from model.vanilla_transformer import VanillaTransformerEncoder
from model.xai_guided_transformer import XAIGuidedTransformerEncoder
from model.losses import XAIGuidedLoss

class Trainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 🧠 Dimensions du modèle
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_heads = config["model"]["num_heads"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout"]
        self.alpha = config["loss"].get("alpha", 0.5)

        # 📦 Modules principaux
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
        self.loss_fn = XAIGuidedLoss(alpha=self.alpha)

        # 🔧 Projections (pour Q et K)
        self.W_q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_k = nn.Linear(self.embedding_dim, self.embedding_dim)

        # 🧮 Classification heads
        self.classifier_coarse = nn.Linear(self.embedding_dim, config["model"]["num_classes"])
        self.classifier_refined = nn.Linear(self.embedding_dim, config["model"]["num_classes"])

    def forward(self, x_raw, freq_bounds, age, gender, y_true=None):
        """
        Args:
            x_raw: Tensor [B, Freq, d] — input tokens from Connectome Tokenizer
            freq_bounds: [Freq, 2]
            age: [B, 1]
            gender: [B, 1]
            y_true: [B] or None — labels (optional)
        Returns:
            If y_true is None: returns logits
            Else: returns loss
        """

        # 🎯 1. dRoFE → encode Q & K avec fréquence, âge, genre
        q = self.W_q(x_raw)
        k = self.W_k(x_raw)

        q_rot = self.drofe(q, freq_bounds, age, gender)
        k_rot = self.drofe(k, freq_bounds, age, gender)

        # 🎯 2. Vanilla Transformer (coarse attention)
        x_coarse = self.vanilla_transformer(q_rot)  # ou utiliser une combinaison q+k
        logits_coarse = self.classifier_coarse(x_coarse.mean(dim=1))  # pooling moyen sur les tokens

        #########################################
        # ⚠️ 3. EXPLAINER → DeepLIFT sur x_coarse pour obtenir Qexpl, Kexpl
        # TODO: appeler explainer ici pour obtenir :
        # q_expl = explainer(x_coarse, target="q")
        # k_expl = explainer(x_coarse, target="k")
        # Placeholder temporaire :
        q_expl = q_rot.detach()
        k_expl = k_rot.detach()
        ###########################################

        # 🎯 4. XAI-guided Transformer (refined attention)
        x_refined = self.xai_transformer(x_raw, q_expl, k_expl)
        logits_refined = self.classifier_refined(x_refined.mean(dim=1))

        # 🎯 5. Perte supervisée (si labels fournis)
        if y_true is not None:
            return self.loss_fn(logits_coarse, logits_refined, y_true)
        else:
            return logits_coarse, logits_refined



"""
Trainer/
│
├── Inputs:
│   ├── x_raw        : Tensor [B, Freq, d]      # Token embeddings from Connectome Tokenizer (GNN output)
│   ├── freq_bounds  : Tensor [Freq, 2]         # EEG frequency band bounds: [f_l, f_u]
│   ├── age          : Tensor [B, 1]            # Patient ages (normalized)
│   ├── gender       : Tensor [B, 1]            # Patient genders (binary or scalar)
│   └── y_true       : Tensor [B] or None       # Ground truth labels (optional)
│
├── Step 1 – dRoFE Encoding:
│   ├── q = W_q(x_raw), k = W_k(x_raw)          # Linear projections
│   ├── q_rot = dRoFE(q, freq_bounds, age, gender)
│   └── k_rot = dRoFE(k, freq_bounds, age, gender)
│
├── Step 2 – Vanilla Transformer Encoder:
│   ├── x_coarse = VanillaTransformer(q_rot)
│   └── logits_coarse = Classifier(mean(x_coarse))     # [B, C]
│
├── Step 3 – XAI Explainer (TODO):
│   ├── q_expl = DeepLIFT(x_coarse)    # Feature importance on Q (placeholder: q_rot.detach())
│   └── k_expl = DeepLIFT(x_coarse)    # Feature importance on K (placeholder: k_rot.detach())
│
├── Step 4 – XAI-Guided Transformer Encoder:
│   ├── x_refined = XAITransformer(x_raw, q_expl, k_expl)
│   └── logits_refined = Classifier(mean(x_refined))   # [B, C]
│
├── Step 5 – Loss Computation (optional):
│   └── total_loss = (1 - α) * CE(logits_coarse, y_true) + α * CE(logits_refined, y_true)
│
└── Outputs:
    ├── If y_true is None:
    │     └── logits_coarse: [B, C]
    │     └── logits_refined: [B, C]
    └── Else:
          └── total_loss: scalar
"""