import torch
import torch.nn as nn
import math

class dRoFEEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        # θ_t = 4πt / d  — angles prédéfinis pour la rotation
        self.theta = 4 * math.pi * torch.arange(0, embedding_dim // 2) / embedding_dim

    def forward(self, x, freq_bounds, age, gender):
        """
        Applique la rotation dRoFE sur les embeddings d'entrée.

        Args:
            x: Tensor [B, Freq, d]         — embeddings (Query ou Key)
            freq_bounds: Tensor [Freq, 2]  — bornes [f_l, f_u] des bandes EEG
            age: Tensor [B, 1]             — âge du patient (normalisé entre 0 et 1)
            gender: Tensor [B, 1]          — genre (ex: 0 pour H, 1 pour F)

        Returns:
            Tensor [B, Freq, d]            — embeddings tournés
        """
        B, Freq, d = x.shape
        device = x.device
        theta = self.theta.to(device)  # [d/2]

        # Extraire f_l et f_u
        f_l = freq_bounds[:, 0].unsqueeze(-1)  # [Freq, 1]
        f_u = freq_bounds[:, 1].unsqueeze(-1)  # [Freq, 1]

        # Calcul des angles de rotation : f * θ_t
        angle_l = f_l * theta  # [Freq, d/2]
        angle_u = f_u * theta  # [Freq, d/2]

        # Calcul des composantes réelles et imaginaires du vecteur de rotation
        rot_real = age.unsqueeze(1) * torch.cos(angle_l) + gender.unsqueeze(1)  # [B, Freq, d/2]
        rot_imag = age.unsqueeze(1) * torch.sin(angle_u) + gender.unsqueeze(1)  # [B, Freq, d/2]

        # Séparer les dimensions paires (réelles) et impaires (imag)
        x1 = x[..., ::2]  # [B, Freq, d/2]
        x2 = x[..., 1::2] # [B, Freq, d/2]

        # Multiplication complexe : R · x
        x_rot_real = rot_real * x1 - rot_imag * x2
        x_rot_imag = rot_real * x2 + rot_imag * x1

        # Réassemblage du vecteur : concat(x_rot_real, x_rot_imag)
        x_rot = torch.stack([x_rot_real, x_rot_imag], dim=-1).flatten(-2)  # [B, Freq, d]

        return x_rot
    
"""
dRoFEEmbedding/
│
├── Inputs:
│   ├── x         : Tensor [B, Freq, d]        # Raw token embeddings (Query or Key)
│   ├── freq_bounds: Tensor [Freq, 2]          # EEG band bounds [f_l, f_u]
│   ├── age       : Tensor [B, 1]              # Patient's age (normalized)
│   └── gender    : Tensor [B, 1]              # Patient's gender (e.g., 0 or 1)
│
├── Step-by-step:
│   ├── Compute θ_t = 4πt/d  ∀ t ∈ {0, ..., d/2 - 1}
│   ├── Compute phase angles:
│   │     - angle_l = f_l * θ_t
│   │     - angle_u = f_u * θ_t
│   ├── Compute rotary weights (Euler’s formula + demographics):
│   │     - Re(R) = age * cos(f_l θ_t) + gender
│   │     - Im(R) = age * sin(f_u θ_t) + gender
│   ├── Split x = [x_even, x_odd] → pairs of dimensions
│   └── Apply complex rotation:
│         x_rot = R · x  (in ℝᵈ via real-imag parts)
│
└── Output:
    └── x_rot: Tensor [B, Freq, d]             # Rotated embeddings (frequency- and demography-aware)
"""