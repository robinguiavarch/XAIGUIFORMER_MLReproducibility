"""
src/model/transformer_used_two_times.py - CORRIGÉ pour Captum avec noms originaux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTransformerLayer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Projections Q, K, V
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # GeGLU feedforward
        self.ff_linear1 = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.ff_linear2 = nn.Linear(embedding_dim, embedding_dim)
        
        # RMSNorm
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.norm2 = nn.RMSNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # ✅ dRoFE intégré (pas externe)
        from model.dRoFE_embedding import dRoFEEmbedding
        self.drofe = dRoFEEmbedding(embedding_dim)

    def geglu(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1) * x2

    def forward(self, x, context_info=None):
        """
        ✅ CAPTUM-COMPATIBLE : Un seul argument tensor + context optionnel
        
        Args:
            x: [B, Freq, d] - embeddings principaux
            context_info: [B, Freq+2, extra] - contexte (freq_bounds + demographics)
                         ou None pour mode XAI-guided
        """
        B, Freq, d = x.shape
        
        # ✅ PROJECTIONS Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # ✅ APPLIQUER dRoFE si context disponible (mode standard)
        if context_info is not None:
            # Parser le context: [B, Freq+2, X] -> freq_bounds + demographics
            freq_bounds = context_info[:, :Freq, :2]  # [B, Freq, 2]
            demographics = context_info[:, Freq:Freq+2, :1].squeeze(-1)  # [B, 2]
            
            # Prendre les freq_bounds du premier échantillon (identiques)
            freq_bounds_single = freq_bounds[0]  # [Freq, 2]
            age = demographics[:, 0:1]  # [B, 1]
            gender = demographics[:, 1:2]  # [B, 1]
            
            # Appliquer dRoFE
            q_rot = self.drofe(q, freq_bounds_single, age, gender)
            k_rot = self.drofe(k, freq_bounds_single, age, gender)
        else:
            # Mode XAI-guided : pas de dRoFE
            q_rot = q
            k_rot = k
        
        # ✅ MULTI-HEAD ATTENTION
        q_rot = q_rot.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        k_rot = k_rot.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Freq, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Freq, d)
        attn_output = self.out_proj(attn_output)
        
        # ✅ RESIDUAL + NORM
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # ✅ FEEDFORWARD (GeGLU)
        ff_out = self.ff_linear2(self.geglu(self.ff_linear1(x)))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class SharedTransformerEncoder(nn.Module):
    """
    ✅ TRANSFORMER COMPATIBLE CAPTUM avec context packing - NOM ORIGINAL
    """
    def __init__(self, embedding_dim=128, num_heads=4, num_layers=12, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SharedTransformerLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.RMSNorm(embedding_dim)
        
        print(f"✅ SharedTransformerEncoder: {num_layers} layers - CAPTUM COMPATIBLE")

    def _pack_context(self, freq_bounds, demographic_info):
        """
        ✅ PACKER freq_bounds + demographics en un seul tensor pour Captum
        """
        B = demographic_info.shape[0]
        Freq = freq_bounds.shape[0]
        device = freq_bounds.device
        
        # Répéter freq_bounds pour chaque échantillon du batch
        freq_bounds_batched = freq_bounds.unsqueeze(0).repeat(B, 1, 1)  # [B, Freq, 2]
        
        # Ajouter demographics en [B, 2, 2] (répéter pour avoir même dernière dim)
        demographics_padded = demographic_info.unsqueeze(-1).repeat(1, 1, 2)  # [B, 2, 2]
        
        # Concaténer : [B, Freq+2, 2]
        context_info = torch.cat([freq_bounds_batched, demographics_padded], dim=1)
        
        return context_info

    def forward(self, x, freq_bounds=None, demographic_info=None, explanations=None, mode='standard'):
        """
        ✅ FORWARD CAPTUM-COMPATIBLE avec interface originale
        
        Args:
            x: [B, Freq, d] - embeddings
            freq_bounds: [Freq, 2] (pour mode standard)
            demographic_info: [B, 2] (pour mode standard)
            explanations: List[Tensor] (pour mode XAI-guided)
            mode: 'standard' ou 'xai_guided'
        """
        context_info = None
        
        if mode == 'standard' and freq_bounds is not None and demographic_info is not None:
            # ✅ PACKER le contexte en un seul tensor
            context_info = self._pack_context(freq_bounds, demographic_info)
        
        # ✅ FORWARD à travers toutes les couches
        for i, layer in enumerate(self.layers):
            if mode == 'xai_guided' and explanations is not None and i < len(explanations):
                # Utiliser explanation comme input alternatif
                x_input = explanations[i]
            else:
                x_input = x
                
            x = layer(x_input, context_info)
        
        return self.final_norm(x)


class CaptumExplainerWrapper(nn.Module):
    """
    ✅ WRAPPER CAPTUM pour l'explainer - utilisé par explainers.py
    """
    def __init__(self, transformer, classifier):
        super().__init__()
        self.transformer = transformer
        self.classifier = classifier

    def forward(self, x, context_info=None):
        """
        ✅ FORWARD CAPTUM : SEULEMENT des tensors
        """
        # Déterminer le mode
        mode = 'standard' if context_info is not None else 'xai_guided'
        
        # Unpacker context si nécessaire
        freq_bounds = None
        demographic_info = None
        
        if context_info is not None:
            B, total_len, _ = context_info.shape
            Freq = total_len - 2
            
            freq_bounds = context_info[0, :Freq, :2]  # [Freq, 2] (premier échantillon)
            demographics_raw = context_info[:, Freq:Freq+2, 0]  # [B, 2]
            demographic_info = demographics_raw  # [B, 2]
        
        # Forward transformer
        x_out = self.transformer(x, freq_bounds, demographic_info, None, mode)
        
        # Classification
        logits = self.classifier(x_out.mean(dim=1))
        
        return logits