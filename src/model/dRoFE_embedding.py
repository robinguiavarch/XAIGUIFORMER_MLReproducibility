import torch
import torch.nn as nn
import math

class dRoFEEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        # θ_t = 4πt / d — predefined rotation angles
        self.theta = 4 * math.pi * torch.arange(0, embedding_dim // 2) / embedding_dim

    def forward(self, x, freq_bounds, age, gender):
        """
        Applies dRoFE rotation to the input embeddings.

        Args:
            x (Tensor): [B, Freq, d]         — input embeddings (Query or Key)
            freq_bounds (Tensor): [Freq, 2]  — EEG band boundaries [f_l, f_u]
            age (Tensor): [B, 1]             — normalized patient age (0 to 1)
            gender (Tensor): [B, 1]          — gender (e.g., 0 for male, 1 for female)

        Returns:
            Tensor: [B, Freq, d] — rotated embeddings
        """
        B, Freq, d = x.shape
        device = x.device
        theta = self.theta.to(device)  # [d/2]

        # Extract lower and upper frequency bounds
        f_l = freq_bounds[:, 0].unsqueeze(-1)  # [Freq, 1]
        f_u = freq_bounds[:, 1].unsqueeze(-1)  # [Freq, 1]

        # Compute phase angles: f * θ_t
        angle_l = f_l * theta  # [Freq, d/2]
        angle_u = f_u * theta  # [Freq, d/2]

        # Compute real and imaginary parts of the rotation vector
        rot_real = age.unsqueeze(1) * torch.cos(angle_l) + gender.unsqueeze(1)  # [B, Freq, d/2]
        rot_imag = age.unsqueeze(1) * torch.sin(angle_u) + gender.unsqueeze(1)  # [B, Freq, d/2]

        # Split embedding into even and odd dimensions (real and imaginary parts)
        x1 = x[..., ::2]  # [B, Freq, d/2]
        x2 = x[..., 1::2] # [B, Freq, d/2]

        # Apply complex rotation: R · x
        x_rot_real = rot_real * x1 - rot_imag * x2
        x_rot_imag = rot_real * x2 + rot_imag * x1

        # Reassemble rotated embedding: concat(x_rot_real, x_rot_imag)
        x_rot = torch.stack([x_rot_real, x_rot_imag], dim=-1).flatten(-2)  # [B, Freq, d]

        return x_rot


"""
dRoFEEmbedding/
│
├── Inputs:
│   ├── x          : Tensor [B, Freq, d]        # Raw token embeddings (Query or Key)
│   ├── freq_bounds: Tensor [Freq, 2]           # EEG band boundaries [f_l, f_u]
│   ├── age        : Tensor [B, 1]              # Patient age (normalized)
│   └── gender     : Tensor [B, 1]              # Patient gender (e.g., 0 or 1)
│
├── Step-by-step:
│   ├── Compute θ_t = 4πt/d  for t ∈ {0, ..., d/2 - 1}
│   ├── Compute phase angles:
│   │     - angle_l = f_l * θ_t
│   │     - angle_u = f_u * θ_t
│   ├── Compute rotary weights (Euler’s formula + demographic modulation):
│   │     - Re(R) = age * cos(f_l * θ_t) + gender
│   │     - Im(R) = age * sin(f_u * θ_t) + gender
│   ├── Split x = [x_even, x_odd] → interpret as real-imaginary pairs
│   └── Apply complex rotation:
│         x_rot = R · x  (in ℝᵈ via real-imag part recombination)
│
└── Output:
    └── x_rot: Tensor [B, Freq, d]              # Rotated embeddings (frequency- and demography-aware)
"""
