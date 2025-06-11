# src/model/explainers.py

import torch
import torch.nn as nn
from captum.attr import DeepLift


class ForwardWrapper(nn.Module):
    """
    Wrapper autour du forward simplifié à utiliser avec Captum.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.target_layer = model.vanilla_transformer
        self.classifier = model.classifier_coarse

    def forward(self, x_input, freq_bounds=None, age=None, gender=None):
        """
        Args:
            x_input: Tensor [B, Freq, d]
        Returns:
            logits: Tensor [B, C]
        """
        x_out = self.target_layer(x_input)
        pooled = x_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class DeepLiftExplainer:
    """
    Explainer XAI basé sur Captum DeepLIFT.
    Applique DeepLIFT sur la sortie du VanillaTransformer.
    """

    def __init__(self, model):
        """
        Args:
            model: Instance du modèle XaiGuiFormer complet
        """
        self.model = model
        self.forward_module = ForwardWrapper(model)
        self.explainer = DeepLift(self.forward_module)

    def get_explanations(self, input_tensor, target, freq_bounds=None, age=None, gender=None):
        """
        Args:
            input_tensor : Tensor [B, Freq, d]
            target       : Tensor [B]
            freq_bounds  : Tensor [Freq, 2]
            age          : Tensor [B, 1]
            gender       : Tensor [B, 1]

        Returns:
            attributions : Tensor [B, Freq, d]
        """
        input_tensor = input_tensor.detach().clone().requires_grad_(True)

        attributions = self.explainer.attribute(
            inputs=input_tensor,
            target=target,
            additional_forward_args=(freq_bounds, age, gender)
        )

        return attributions


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