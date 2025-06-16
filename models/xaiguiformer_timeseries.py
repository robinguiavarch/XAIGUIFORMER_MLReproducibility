import torch
import torch.nn as nn
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules with error handling
try:
    from modules.multirocket_tokenizer import MultiROCKETTokenizer
except Exception as e:
    print(f"Failed to import MultiROCKETTokenizer: {e}")
    sys.exit(1)

try:
    from modules.transformer import XAIguiTransformerEncoder
except Exception as e:
    print(f"Failed to import XAIguiTransformerEncoder: {e}")
    sys.exit(1)

try:
    from modules.explainer import Explainer
except Exception as e:
    print(f"Failed to import Explainer: {e}")
    sys.exit(1)

try:
    from utils.visualizer import get_local
except Exception as e:
    print(f"Failed to import get_local: {e}")
    sys.exit(1)


class XAIguiFormerTimeSeries(nn.Module):
    """
    XAI-guided Transformer for EEG Time Series Classification.

    Replaces the ConnectomeEncoder + GNN pipeline with MultiROCKET tokenization
    while preserving the XAI-guided transformer architecture and explainability features.
    """

    def __init__(
        self,
        num_channels: int = 33,
        num_classes: int = 4,
        freqband: torch.Tensor = None,
        num_kernels: int = 200,
        attention_heads: int = 2,
        output_features: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 12,
        explainer_type: str = 'DeepLift',
        mlp_ratio: float = 4.0,
        act_func=nn.GELU,
        norm=nn.LayerNorm,
        layer_norm_eps: float = 1e-05,
        bias: bool = True,
        dropout: float = 0.1,
        init_values: Optional[float] = None,
        attn_drop: float = 0.0,
        droppath: float = 0.0,
        dim_feedforward: Optional[int] = None
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.output_features = output_features

        if freqband is None or freqband.shape != (9, 2):
            raise ValueError("freqband tensor must be provided with shape [9, 2]")

        self.freqband = freqband
        num_frequency_bands = freqband.shape[0]

        self.multirocket_tokenizer = MultiROCKETTokenizer(
            num_frequency_bands=num_frequency_bands,
            num_channels=num_channels,
            output_features=output_features,
            num_kernels=num_kernels,
            attention_heads=attention_heads,
            dropout=dropout
        )

        self.transformer_encoder = XAIguiTransformerEncoder(
            in_features=output_features,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            freqband=freqband,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            act_func=act_func,
            layer_norm_eps=layer_norm_eps,
            mlp_ratio=mlp_ratio,
            bias=bias,
            init_values=init_values,
            attn_drop=attn_drop,
            droppath=droppath,
            norm=norm
        )

        self.explainer = Explainer(
            model=self.transformer_encoder,
            layer=self.transformer_encoder.TransformerEncoder,
            explainer_type=explainer_type
        )

    @get_local('contribution')
    def forward(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass of XAIguiFormerTimeSeries.

        Args:
            batch: Dictionary with keys 'frequency_tokens', 'demographic_info', optionally 'y' and 'eid'

        Returns:
            List[torch.Tensor]: [vanilla_predictions, xai_refined_predictions]
        """
        frequency_tokens = batch['frequency_tokens']
        demographic_info = batch['demographic_info']

        batch_size = frequency_tokens.shape[0]
        expected_shape = (batch_size, 9, self.num_channels)

        if frequency_tokens.shape[:3] != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {frequency_tokens.shape[:3]}")

        if demographic_info.shape != (batch_size, 2):
            raise ValueError(f"Expected demographic_info shape ({batch_size}, 2), got {demographic_info.shape}")

        device = frequency_tokens.device
        demographic_info = demographic_info.to(device)
        self.freqband = self.freqband.to(device)

        freq_series = self.multirocket_tokenizer(frequency_tokens, demographic_info)
        pred_y_vanilla = self.transformer_encoder(freq_series, demographic_info)
        target_classes = pred_y_vanilla.argmax(dim=1)

        explanation = self.explainer(
            x=(freq_series, demographic_info),
            target=target_classes
        )

        pred_y_refined = self.transformer_encoder(freq_series, demographic_info, explanation)
        return [pred_y_vanilla, pred_y_refined]

    def get_device(self) -> torch.device:
        """Return model device."""
        return next(self.parameters()).device

    def get_attention_weights(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract MultiROCKET attention weights.

        Args:
            batch: Input dictionary

        Returns:
            torch.Tensor: Attention weights
        """
        return self.multirocket_tokenizer.get_attention_weights(batch['frequency_tokens'])

    def extract_freq_series(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract frequency series representation.

        Args:
            batch: Input dictionary

        Returns:
            torch.Tensor: Frequency series
        """
        with torch.no_grad():
            return self.multirocket_tokenizer(
                batch['frequency_tokens'],
                batch['demographic_info']
            )

def get_frequency_bands_tensor():
    """
    Return frequency band tensor.

    Returns:
        torch.Tensor: Frequency bands [9, 2]
    """
    frequency_bands = {
        'delta': [2., 4.],
        'theta': [4., 8.],
        'low_alpha': [8., 10.],
        'high_alpha': [10., 12.],
        'low_beta': [12., 18.],
        'mid_beta': [18., 21.],
        'high_beta': [21., 30.],
        'gamma': [30., 45.],
        'beta': [12., 30.]
    }
    return torch.tensor(list(frequency_bands.values()), dtype=torch.float32)
