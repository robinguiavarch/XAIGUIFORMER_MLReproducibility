# src/model/explainers.py

import torch
import torch.nn as nn
from captum.attr import DeepLift

class Explainer:
    """
    XAI Explainer using DeepLIFT from Captum.
    Applies DeepLIFT on the Vanilla Transformer to get token-level importances.

    Used to compute Q_expl and K_expl as required by the XAI-Guided Transformer.
    """

    def __init__(self, model, target_layer="vanilla_transformer"):
        """
        Args:
            model: The full model (Trainer instance) containing vanilla transformer
            target_layer: str, name of the attribute to explain (usually 'vanilla_transformer')
        """
        self.model = model
        self.target_layer = getattr(model, target_layer)
        self.embedding_dim = model.embedding_dim
        self.num_classes = model.classifier_coarse.out_features

        self.explainer = DeepLift(self.forward_fn)

    def forward_fn(self, x_input):
        """
        Isolated forward pass for DeepLift attribution.
        This passes x_input through vanilla transformer → classifier_coarse

        Args:
            x_input: [B, Freq, d]
        Returns:
            logits: [B, C]
        """
        x_out = self.target_layer(x_input)  # [B, Freq, d]
        pooled = x_out.mean(dim=1)          # mean-pooling
        logits = self.model.classifier_coarse(pooled)  # [B, C]
        return logits

    def get_explanations(self, input_tensor, target_labels):
        """
        Applies DeepLIFT to compute attributions on input_tensor.

        Args:
            input_tensor: [B, Freq, d] — input embeddings (e.g., q_rot or k_rot)
            target_labels: [B] — class labels

        Returns:
            attributions: [B, Freq, d]
        """
        # Generate baseline (zero vector of same shape)
        baseline = torch.zeros_like(input_tensor)

        # Compute DeepLIFT attributions
        attributions = self.explainer.attribute(inputs=input_tensor,
                                                baselines=baseline,
                                                target=target_labels)

        return attributions.detach()

""""
Explainer/
│
├── Inputs:
│   ├── x_input      : Tensor [B, Freq, d]       # Token embeddings (e.g., q_rot or k_rot)
│   └── target_labels: Tensor [B]               # Ground-truth class labels
│
├── Internals:
│   ├── forward_fn(x_input)
│   │   ├── Passes x_input through:
│   │   │   → VanillaTransformerEncoder(x_input)
│   │   │   → Mean-pooling over frequency axis: x_mean = x_out.mean(dim=1)
│   │   │   → Linear classifier: classifier_coarse(x_mean)
│   │   └── Returns logits [B, C]
│   │
│   ├── explainer = DeepLift(forward_fn)
│   │   └── Captum DeepLift wrapper around forward_fn
│
├── Step-by-step:
│   ├── 1. Construct baseline: zeros of shape [B, Freq, d]
│   ├── 2. Compute DeepLIFT attributions:
│   │     attributions = DeepLift.attribute(inputs=x_input, baselines=0, target=target_labels)
│
└── Outputs:
    └── attributions: Tensor [B, Freq, d]        # Token-level importance maps
"""