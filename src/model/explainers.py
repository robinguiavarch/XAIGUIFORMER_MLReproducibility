"""
src/model/explainers.py - Multi-layer Captum-compatible explainer for XaiGuiFormer
"""

import torch
import torch.nn as nn
from captum.attr import LayerDeepLift


class MultiLayerDeepLiftExplainer:
    """
    Multi-layer DeepLIFT explainer compatible with Captum for transformer-based models.

    This class wraps each layer of the shared transformer using Captum's LayerDeepLift,
    allowing layer-wise attribution generation. Input formatting is adapted for 
    XaiGuiFormer's dual-input structure (embeddings + context).
    """

    def __init__(self, model):
        """
        Initialize the multi-layer explainer with Captum wrappers.

        Args:
            model (nn.Module): The XaiGuiFormer model instance to explain.
        """
        self.model = model

        # Import Captum-compatible wrapper for the shared transformer block
        from model.transformer_used_two_times import CaptumExplainerWrapper

        # Wrap the shared transformer for Captum compatibility
        self.captum_wrapper = CaptumExplainerWrapper(
            model.shared_transformer,
            model.classifier_coarse
        )

        # Create a LayerDeepLift explainer for each transformer layer
        self.layer_explainers = []
        for i, layer in enumerate(model.shared_transformer.layers):
            try:
                explainer = LayerDeepLift(
                    self.captum_wrapper,
                    layer,
                    multiply_by_inputs=True
                )
                self.layer_explainers.append(explainer)
                print(f"Layer {i} explainer created.")
            except Exception as e:
                print(f"Layer {i} explainer failed: {e}")
                self.layer_explainers.append(None)

        valid_expl = sum(e is not None for e in self.layer_explainers)
        print(f"MultiLayerDeepLiftExplainer initialized with {valid_expl}/{len(self.layer_explainers)} valid layers.")

    def _pack_inputs_for_captum(self, x_embeddings, freq_bounds, demographic_info):
        """
        Format inputs for Captum's multi-input interface.

        Args:
            x_embeddings (Tensor): [B, Freq, d] - token embeddings.
            freq_bounds (Tensor): [Freq, 2] - frequency band boundaries.
            demographic_info (Tensor): [B, 2] - age and gender information.

        Returns:
            Tuple[Tensor, Tensor]: Formatted inputs (x_embeddings, context_info).
        """
        B = x_embeddings.shape[0]
        Freq = freq_bounds.shape[0]

        freq_bounds_batched = freq_bounds.unsqueeze(0).repeat(B, 1, 1)        # [B, Freq, 2]
        demographics_padded = demographic_info.unsqueeze(-1).repeat(1, 1, 2)  # [B, 2, 2]

        # Concatenate frequency and demographic information
        context_info = torch.cat([freq_bounds_batched, demographics_padded], dim=1)  # [B, Freq + 2, 2]

        return x_embeddings, context_info

    def get_explanations(self, x_embeddings, target, freq_bounds, age, gender):
        """
        Compute per-layer feature attributions using DeepLIFT.

        Args:
            x_embeddings (Tensor): [B, Freq, d] - input embeddings (Query/Key).
            target (Tensor or int): Target class index or tensor for attribution.
            freq_bounds (Tensor): [Freq, 2] - EEG band limits.
            age (Tensor): [B, 1] - normalized age values.
            gender (Tensor): [B, 1] - gender indicators.

        Returns:
            List[Tensor]: A list of [B, Freq, d] tensors, one per layer.
                          Contains feature attributions or fallback inputs.
        """
        demographic_info = torch.cat([age, gender], dim=1)  # [B, 2]

        x_input, context_info = self._pack_inputs_for_captum(
            x_embeddings, freq_bounds, demographic_info
        )

        baselines = (
            torch.zeros_like(x_input),
            torch.zeros_like(context_info)
        )

        explanations = []

        for i, explainer in enumerate(self.layer_explainers):
            if explainer is None:
                explanations.append(x_input.detach())
                continue

            try:
                attribution = explainer.attribute(
                    inputs=(x_input, context_info),
                    baselines=baselines,
                    target=target,
                    attribute_to_layer_input=True
                )

                if isinstance(attribution, tuple):
                    attribution = attribution[0]

                explanations.append(attribution.detach())
                print(f"Layer {i} explanation generated with shape: {attribution.shape}")
            except Exception as e:
                print(f"Layer {i} explanation failed: {e}")
                explanations.append(x_input.detach())

        print(f"Generated explanations for {len(explanations)} layers.")
        return explanations
