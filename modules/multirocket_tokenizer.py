import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

class MultiROCKETTokenizer(nn.Module):
    """
    MultiROCKET-based tokenizer for EEG frequency bands.

    Transforms frequency tokens [batch, bands, channels, time] 
    into transformer-compatible sequences [batch, bands, features].
    """

    def __init__(
        self,
        num_frequency_bands: int = 9,
        num_channels: int = 33,
        output_features: int = 128,
        num_kernels: int = 5000,
        kernel_sizes: list = [7, 9, 11],
        dilation_sizes: list = [1, 2, 3],
        attention_heads: int = 8,
        dropout: float = 0.1,
        seed: int = 42
    ):
        super().__init__()

        self.num_frequency_bands = num_frequency_bands
        self.num_channels = num_channels
        self.output_features = output_features
        self.num_kernels = num_kernels
        self.kernel_sizes = kernel_sizes
        self.dilation_sizes = dilation_sizes
        self.attention_heads = attention_heads

        self.rocket_features_per_series = num_kernels * 2

        rocket_kernels = self._generate_rocket_kernels(seed)
        self.register_buffer('rocket_kernels', rocket_kernels)

        self.channel_attention = nn.MultiheadAttention(
            embed_dim=self.rocket_features_per_series,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(self.rocket_features_per_series)
        self.rocket_norm = nn.BatchNorm1d(self.rocket_features_per_series)

        self.projection = nn.Sequential(
            nn.Linear(self.rocket_features_per_series, output_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_features * 2, output_features)
        )

        self._init_weights()

    def _generate_rocket_kernels(self, seed: int) -> torch.Tensor:
        """
        Generate a fixed number of 1D convolution kernels with random weights
        using specified sizes and dilations. Ensures all kernels are padded to the
        same maximum length for batch processing.

        Args:
            seed (int): Random seed for reproducibility.

        Returns:
            torch.Tensor: Tensor of shape [num_kernels, 1, kernel_length] containing all padded kernels.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        kernels = []
        kernels_per_size = self.num_kernels // len(self.kernel_sizes)

        for kernel_size in self.kernel_sizes:
            for dilation in self.dilation_sizes:
                num_kernels_combo = kernels_per_size // len(self.dilation_sizes)

                for _ in range(num_kernels_combo):
                    kernel_weights = torch.randn(kernel_size)
                    kernel_weights = kernel_weights / torch.norm(kernel_weights)

                    if dilation > 1:
                        dilated_kernel = torch.zeros(kernel_size + (kernel_size - 1) * (dilation - 1))
                        dilated_kernel[::dilation] = kernel_weights
                        kernel_weights = dilated_kernel

                    kernel = kernel_weights.unsqueeze(0).unsqueeze(0)
                    kernels.append(kernel)

        while len(kernels) < self.num_kernels:
            kernel_size = np.random.choice(self.kernel_sizes)
            kernel_weights = torch.randn(kernel_size)
            kernel_weights = kernel_weights / torch.norm(kernel_weights)
            kernel = kernel_weights.unsqueeze(0).unsqueeze(0)
            kernels.append(kernel)

        max_kernel_size = max(k.shape[-1] for k in kernels)
        padded_kernels = []
        for kernel in kernels:
            current_size = kernel.shape[-1]
            if current_size < max_kernel_size:
                pad_left = (max_kernel_size - current_size) // 2
                pad_right = max_kernel_size - current_size - pad_left
                padded_kernel = F.pad(kernel, (pad_left, pad_right), mode='constant', value=0)
            else:
                padded_kernel = kernel
            padded_kernels.append(padded_kernel)

        all_kernels = torch.cat(padded_kernels[:self.num_kernels], dim=0)
        return all_kernels

    def _init_weights(self):
        """
        Initialize the weights of the projection layers using Xavier initialization.
        Bias terms are set to zero if present.
        """
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extract_rocket_features(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Extract ROCKET features from raw time series input using 1D convolutions.

        Args:
            time_series (torch.Tensor): Tensor of shape [batch_size * num_bands * num_channels, time_points].

        Returns:
            torch.Tensor: Extracted features of shape [batch_size * num_bands * num_channels, num_kernels * 2].
        """
        batch_series, time_points = time_series.shape
        device = time_series.device
        conv_input = time_series.unsqueeze(1)
        all_features = []
        chunk_size = 1000

        for i in range(0, self.num_kernels, chunk_size):
            end_idx = min(i + chunk_size, self.num_kernels)
            kernel_chunk = self.rocket_kernels[i:end_idx]

            conv_output = F.conv1d(
                conv_input, 
                kernel_chunk, 
                padding='same',
                groups=1
            )

            max_features = torch.max(conv_output, dim=2)[0]
            mean_features = torch.mean(conv_output, dim=2)

            chunk_features = torch.stack([max_features, mean_features], dim=2)
            chunk_features = chunk_features.reshape(batch_series, -1)

            all_features.append(chunk_features)

        rocket_features = torch.cat(all_features, dim=1)
        return rocket_features

    def _aggregate_channels_with_attention(self, rocket_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Aggregates extracted ROCKET features across EEG channels for each frequency band
        using multi-head self-attention.

        Args:
            rocket_features (torch.Tensor): Flattened ROCKET features of shape [batch_size * bands * channels, features].
            batch_size (int): Number of samples in the batch.

        Returns:
            torch.Tensor: Aggregated features per band of shape [batch_size, bands, features].
        """
        features_per_band = rocket_features.reshape(
            batch_size * self.num_frequency_bands, 
            self.num_channels, 
            self.rocket_features_per_series
        )

        features_per_band = self.attention_norm(features_per_band)
        attended_features, _ = self.channel_attention(
            features_per_band,
            features_per_band,
            features_per_band
        )
        band_features = torch.mean(attended_features, dim=1)

        aggregated_features = band_features.reshape(
            batch_size, 
            self.num_frequency_bands, 
            self.rocket_features_per_series
        )

        return aggregated_features

    def forward(self, frequency_tokens: torch.Tensor, demographic_info: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiROCKETTokenizer.

        Args:
            frequency_tokens (torch.Tensor): Input tensor of shape [batch, bands, channels, time].
            demographic_info (torch.Tensor): Unused in this module but kept for compatibility.

        Returns:
            torch.Tensor: Projected feature representation of shape [batch, bands, output_features].
        """
        batch_size, num_bands, num_channels, time_points = frequency_tokens.shape

        assert num_bands == self.num_frequency_bands
        assert num_channels == self.num_channels

        time_series = frequency_tokens.reshape(-1, time_points)
        rocket_features = self._extract_rocket_features(time_series)
        rocket_features = self.rocket_norm(rocket_features)
        aggregated_features = self._aggregate_channels_with_attention(rocket_features, batch_size)
        freq_series = self.projection(aggregated_features)

        return freq_series

    def get_attention_weights(self, frequency_tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns attention weights used to aggregate channel features for each band.

        Args:
            frequency_tokens (torch.Tensor): Input tensor of shape [batch, bands, channels, time].

        Returns:
            torch.Tensor: Attention weights of shape [batch, bands, heads, channels, channels].
        """
        batch_size, num_bands, num_channels, time_points = frequency_tokens.shape

        time_series = frequency_tokens.reshape(-1, time_points)
        rocket_features = self._extract_rocket_features(time_series)
        rocket_features = self.rocket_norm(rocket_features)

        features_per_band = rocket_features.reshape(
            batch_size * num_bands, 
            num_channels, 
            self.rocket_features_per_series
        )
        features_per_band = self.attention_norm(features_per_band)

        _, attention_weights = self.channel_attention(
            features_per_band, 
            features_per_band, 
            features_per_band
        )

        attention_weights = attention_weights.reshape(
            batch_size, num_bands, self.attention_heads, num_channels, num_channels
        )

        return attention_weights