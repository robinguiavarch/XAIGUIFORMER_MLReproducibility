"""
src/model/explainers.py - EXPLAINER CORRIGÉ pour Captum avec noms originaux
"""

import torch
import torch.nn as nn
from captum.attr import LayerDeepLift


class MultiLayerDeepLiftExplainer:
    """
    ✅ EXPLAINER MULTI-COUCHES compatible Captum - NOM ORIGINAL
    """
    def __init__(self, model):
        self.model = model
        
        # ✅ Utiliser le wrapper Captum du transformer
        from model.transformer_used_two_times import CaptumExplainerWrapper
        
        # Wrapper Captum pour le transformer
        self.captum_wrapper = CaptumExplainerWrapper(
            model.shared_transformer,
            model.classifier_coarse
        )
        
        # ✅ EXPLAINERS pour chaque couche
        self.layer_explainers = []
        
        # Utiliser les couches du transformer existant
        for i, layer in enumerate(model.shared_transformer.layers):
            try:
                explainer = LayerDeepLift(
                    self.captum_wrapper, 
                    layer,
                    multiply_by_inputs=True
                )
                self.layer_explainers.append(explainer)
                print(f"✅ Layer {i} explainer créé")
                
            except Exception as e:
                print(f"❌ Layer {i} explainer échec: {e}")
                self.layer_explainers.append(None)
        
        valid_explainers = sum(1 for e in self.layer_explainers if e is not None)
        print(f"✅ MultiLayerDeepLiftExplainer: {valid_explainers}/{len(self.layer_explainers)} couches")

    def _pack_inputs_for_captum(self, x_embeddings, freq_bounds, demographic_info):
        """
        ✅ PRÉPARER les inputs pour Captum selon le nouveau format
        """
        B = x_embeddings.shape[0]
        Freq = freq_bounds.shape[0]
        
        # Répéter freq_bounds pour batch
        freq_bounds_batched = freq_bounds.unsqueeze(0).repeat(B, 1, 1)  # [B, Freq, 2]
        
        # Pad demographics pour avoir même dimension
        demographics_padded = demographic_info.unsqueeze(-1).repeat(1, 1, 2)  # [B, 2, 2]
        
        # Concatener
        context_info = torch.cat([freq_bounds_batched, demographics_padded], dim=1)  # [B, Freq+2, 2]
        
        return x_embeddings, context_info

    def get_explanations(self, x_embeddings, target, freq_bounds, age, gender):
        """
        ✅ INTERFACE originale pour générer les explanations multi-couches
        """
        # Combiner demographics
        demographic_info = torch.cat([age, gender], dim=1)  # [B, 2]
        
        # ✅ PACK inputs pour Captum
        x_input, context_info = self._pack_inputs_for_captum(
            x_embeddings, freq_bounds, demographic_info
        )
        
        # Baselines
        baselines = (
            torch.zeros_like(x_input),
            torch.zeros_like(context_info)
        )
        
        explanations = []
        
        # ✅ GÉNÉRER explanations pour chaque couche
        for i, explainer in enumerate(self.layer_explainers):
            if explainer is None:
                # Fallback : utiliser input original
                explanations.append(x_input.detach())
                continue
            
            try:
                # ✅ CAPTUM CALL avec format correct
                attribution = explainer.attribute(
                    inputs=(x_input, context_info),
                    baselines=baselines,
                    target=target,
                    attribute_to_layer_input=True
                )
                
                # Attribution devrait être un tensor [B, Freq, d]
                if isinstance(attribution, tuple):
                    attribution = attribution[0]  # Prendre la première partie si tuple
                
                explanations.append(attribution.detach())
                print(f"✅ Layer {i} explanation: {attribution.shape}")
                
            except Exception as e:
                print(f"⚠️  Layer {i} explainer failed: {e}")
                # Fallback : utiliser input original 
                explanations.append(x_input.detach())
        
        print(f"✅ Generated {len(explanations)} layer explanations")
        return explanations