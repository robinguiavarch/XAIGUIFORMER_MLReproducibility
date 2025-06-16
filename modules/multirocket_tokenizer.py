import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class MultiROCKETTokenizer(nn.Module):
    """
    MultiROCKET-based tokenizer for EEG frequency bands.
    
    Transforms frequency tokens [batch, 9_bands, 33_channels, time_points] 
    into transformer-compatible sequence [batch, 9, 128] while preserving
    inter-channel dependencies and individual channel patterns.
    
    Args:
        num_frequency_bands: Number of frequency bands (default: 9)
        num_channels: Number of EEG channels (default: 33) 
        output_features: Output feature dimension for transformer (default: 128)
        num_kernels: Number of ROCKET kernels (default: 5000)
        kernel_sizes: Sizes of convolution kernels (default: [7, 9, 11])
        dilation_sizes: Dilation values for kernels (default: [1, 2, 3])
        attention_heads: Number of attention heads for channel aggregation (default: 8)
        dropout: Dropout rate (default: 0.1)
        seed: Random seed for reproducible kernel generation (default: 42)
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
        
        # Calculate total number of features from ROCKET
        # Each kernel produces 2 features (max pooling + mean pooling)
        self.rocket_features_per_series = num_kernels * 2
        
        print(f"🚀 Initializing MultiROCKET Tokenizer:")
        print(f"   📊 Input: [{num_frequency_bands} bands, {num_channels} channels, time_points]")
        print(f"   🔧 ROCKET: {num_kernels} kernels → {self.rocket_features_per_series} features/series")
        print(f"   🧠 Output: [batch, {num_frequency_bands}, {output_features}] for transformer")
        
        # Generate and register ROCKET kernels (non-trainable, reproducible)
        rocket_kernels = self._generate_rocket_kernels(seed)
        self.register_buffer('rocket_kernels', rocket_kernels)
        
        # Learnable channel attention for inter-channel dependencies
        # Input: [33_channels, rocket_features] → Output: [aggregated_features]
        self.channel_attention = nn.MultiheadAttention(
            embed_dim=self.rocket_features_per_series,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm for attention
        self.attention_norm = nn.LayerNorm(self.rocket_features_per_series)
        
        # Batch normalization for ROCKET features stabilization
        self.rocket_norm = nn.BatchNorm1d(self.rocket_features_per_series)
        
        # Final projection to transformer dimension
        self.projection = nn.Sequential(
            nn.Linear(self.rocket_features_per_series, output_features * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_features * 2, output_features)
        )
        
        # Initialize projection weights
        self._init_weights()
        
        print(f"✅ MultiROCKET Tokenizer initialized successfully!")
    
    def _generate_rocket_kernels(self, seed: int) -> torch.Tensor:
        """
        Generate ROCKET convolution kernels with reproducible randomness.
        
        Returns:
            torch.Tensor: Kernels of shape [num_kernels, 1, kernel_length]
        """
        print(f"🔧 Generating {self.num_kernels} ROCKET kernels (seed={seed})...")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        kernels = []
        kernels_per_size = self.num_kernels // len(self.kernel_sizes)
        
        for kernel_size in self.kernel_sizes:
            for dilation in self.dilation_sizes:
                # Number of kernels for this size/dilation combination
                num_kernels_combo = kernels_per_size // len(self.dilation_sizes)
                
                for _ in range(num_kernels_combo):
                    # Generate random weights for kernel
                    kernel_weights = torch.randn(kernel_size)
                    
                    # Normalize kernel
                    kernel_weights = kernel_weights / torch.norm(kernel_weights)
                    
                    # Create dilated kernel if needed
                    if dilation > 1:
                        dilated_kernel = torch.zeros(kernel_size + (kernel_size - 1) * (dilation - 1))
                        dilated_kernel[::dilation] = kernel_weights
                        kernel_weights = dilated_kernel
                    
                    # Reshape for conv1d: [out_channels=1, in_channels=1, kernel_length]
                    kernel = kernel_weights.unsqueeze(0).unsqueeze(0)
                    kernels.append(kernel)
        
        # Fill remaining kernels if needed
        while len(kernels) < self.num_kernels:
            kernel_size = np.random.choice(self.kernel_sizes)
            kernel_weights = torch.randn(kernel_size)
            kernel_weights = kernel_weights / torch.norm(kernel_weights)
            kernel = kernel_weights.unsqueeze(0).unsqueeze(0)
            kernels.append(kernel)
        
        # FIXED: Pad all kernels to uniform size before concatenation (Option A)
        if len(kernels) > 0:
            # Calculate maximum kernel size
            max_kernel_size = max(k.shape[-1] for k in kernels)
            print(f"   Max kernel size: {max_kernel_size}, will pad smaller kernels")
            
            # Pad all kernels to max size
            padded_kernels = []
            for i, kernel in enumerate(kernels):
                current_size = kernel.shape[-1]
                if current_size < max_kernel_size:
                    padding_needed = max_kernel_size - current_size
                    # Symmetric padding: pad_left = padding//2, pad_right = padding - padding//2
                    pad_left = padding_needed // 2
                    pad_right = padding_needed - pad_left
                    # F.pad format: (pad_left, pad_right) for last dimension
                    padded_kernel = torch.nn.functional.pad(kernel, (pad_left, pad_right), mode='constant', value=0)
                else:
                    padded_kernel = kernel
                
                padded_kernels.append(padded_kernel)
                
                # Log first few kernel sizes for debugging
                if i < 5:
                    print(f"     Kernel {i}: {current_size} → {padded_kernel.shape[-1]}")
            
            # Now all kernels have the same size, safe to concatenate
            all_kernels = torch.cat(padded_kernels[:self.num_kernels], dim=0)
        else:
            raise RuntimeError("No kernels generated")
        
        print(f"   Generated {len(kernels)} kernels with shapes: {[k.shape[-1] for k in kernels[:5]]}...")
        
        return all_kernels
    
    def _init_weights(self):
        """Initialize learnable parameters."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _extract_rocket_features(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Extract ROCKET features from time series data.
        
        Args:
            time_series: Input tensor [batch*bands*channels, time_points]
            
        Returns:
            torch.Tensor: ROCKET features [batch*bands*channels, rocket_features]
        """
        batch_series, time_points = time_series.shape
        device = time_series.device
        
        # Prepare input for conv1d: [batch*bands*channels, 1, time_points]
        conv_input = time_series.unsqueeze(1)
        
        # Extract features using all kernels
        all_features = []
        
        # Process kernels in chunks to manage memory
        chunk_size = 1000
        for i in range(0, self.num_kernels, chunk_size):
            end_idx = min(i + chunk_size, self.num_kernels)
            kernel_chunk = self.rocket_kernels[i:end_idx]
            
            # Apply convolutions
            # conv_input: [batch*bands*channels, 1, time_points]
            # kernel_chunk: [chunk_kernels, 1, kernel_size]
            # Output: [batch*bands*channels, chunk_kernels, time_points]
            conv_output = F.conv1d(
                conv_input, 
                kernel_chunk, 
                padding='same',
                groups=1
            )
            
            # Global max and mean pooling for each kernel
            # Max pooling: [batch*bands*channels, chunk_kernels]
            max_features = torch.max(conv_output, dim=2)[0]
            # Mean pooling: [batch*bands*channels, chunk_kernels]  
            mean_features = torch.mean(conv_output, dim=2)
            
            # Interleave max and mean features
            chunk_features = torch.stack([max_features, mean_features], dim=2)
            chunk_features = chunk_features.reshape(batch_series, -1)
            
            all_features.append(chunk_features)
        
        # Concatenate all features: [batch*bands*channels, rocket_features]
        rocket_features = torch.cat(all_features, dim=1)
        
        return rocket_features
    
    def _aggregate_channels_with_attention(
        self, 
        rocket_features: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Aggregate channel features using learnable attention to capture inter-channel dependencies.
        
        Args:
            rocket_features: [batch*bands*channels, rocket_features]
            batch_size: Original batch size
            
        Returns:
            torch.Tensor: Aggregated features [batch, bands, rocket_features]
        """
        # Reshape to separate batch, bands, and channels
        # [batch*bands*channels, rocket_features] → [batch*bands, channels, rocket_features]
        features_per_band = rocket_features.reshape(
            batch_size * self.num_frequency_bands, 
            self.num_channels, 
            self.rocket_features_per_series
        )
        
        # Apply layer normalization
        features_per_band = self.attention_norm(features_per_band)
        
        # Apply multi-head attention across channels for each frequency band
        # Query, Key, Value are all the same (self-attention across channels)
        # Input/Output: [batch*bands, channels, features]
        attended_features, attention_weights = self.channel_attention(
            features_per_band,  # query
            features_per_band,  # key  
            features_per_band   # value
        )
        
        # Global average pooling across channels to get band-level representation
        # [batch*bands, channels, features] → [batch*bands, features]
        band_features = torch.mean(attended_features, dim=1)
        
        # Reshape to final format: [batch*bands, features] → [batch, bands, features]
        aggregated_features = band_features.reshape(
            batch_size, 
            self.num_frequency_bands, 
            self.rocket_features_per_series
        )
        
        return aggregated_features
    
    def forward(
        self, 
        frequency_tokens: torch.Tensor, 
        demographic_info: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of MultiROCKET tokenizer.
        
        Args:
            frequency_tokens: [batch, 9_bands, 33_channels, time_points]
            demographic_info: [batch, 2] - passed through unchanged
            
        Returns:
            torch.Tensor: Tokenized frequency series [batch, 9, 128]
        """
        batch_size, num_bands, num_channels, time_points = frequency_tokens.shape
        
        # Validate input dimensions
        assert num_bands == self.num_frequency_bands, f"Expected {self.num_frequency_bands} bands, got {num_bands}"
        assert num_channels == self.num_channels, f"Expected {self.num_channels} channels, got {num_channels}"
        
        # Step 1: Reshape for parallel ROCKET processing
        # [batch, bands, channels, time] → [batch*bands*channels, time]
        time_series = frequency_tokens.reshape(-1, time_points)
        
        # Step 2: Extract ROCKET features (non-trainable feature extraction)
        # [batch*bands*channels, time] → [batch*bands*channels, rocket_features]
        rocket_features = self._extract_rocket_features(time_series)
        
        # Step 3: Apply batch normalization for stability
        # Reshape for BatchNorm1d: [batch*bands*channels, features]
        rocket_features = self.rocket_norm(rocket_features)
        
        # Step 4: Aggregate channels with learnable attention
        # [batch*bands*channels, rocket_features] → [batch, bands, rocket_features]
        aggregated_features = self._aggregate_channels_with_attention(rocket_features, batch_size)
        
        # Step 5: Project to transformer dimension
        # [batch, bands, rocket_features] → [batch, bands, output_features]
        freq_series = self.projection(aggregated_features)
        
        return freq_series
    
    def get_attention_weights(
        self, 
        frequency_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability.
        
        Args:
            frequency_tokens: [batch, 9_bands, 33_channels, time_points]
            
        Returns:
            torch.Tensor: Attention weights [batch, bands, heads, channels, channels]
        """
        batch_size, num_bands, num_channels, time_points = frequency_tokens.shape
        
        # Extract ROCKET features
        time_series = frequency_tokens.reshape(-1, time_points)
        rocket_features = self._extract_rocket_features(time_series)
        rocket_features = self.rocket_norm(rocket_features)
        
        # Reshape for attention
        features_per_band = rocket_features.reshape(
            batch_size * num_bands, 
            num_channels, 
            self.rocket_features_per_series
        )
        features_per_band = self.attention_norm(features_per_band)
        
        # Get attention weights
        _, attention_weights = self.channel_attention(
            features_per_band, features_per_band, features_per_band
        )
        
        # Reshape: [batch*bands, heads, channels, channels] → [batch, bands, heads, channels, channels]
        attention_weights = attention_weights.reshape(
            batch_size, num_bands, self.attention_heads, num_channels, num_channels
        )
        
        return attention_weights


def test_multirocket_tokenizer():
    """
    Comprehensive test suite for MultiROCKET Tokenizer.
    """
    print("🧪 Testing MultiROCKET Tokenizer...")
    
    # Test parameters
    batch_size = 4
    num_bands = 9
    num_channels = 33
    time_points = 29910
    output_features = 128
    
    # Create test data
    frequency_tokens = torch.randn(batch_size, num_bands, num_channels, time_points)
    demographic_info = torch.randn(batch_size, 2)
    
    print(f"📊 Test input shapes:")
    print(f"   Frequency tokens: {frequency_tokens.shape}")
    print(f"   Demographic info: {demographic_info.shape}")
    
    # Initialize tokenizer
    tokenizer = MultiROCKETTokenizer(
        num_frequency_bands=num_bands,
        num_channels=num_channels,
        output_features=output_features,
        num_kernels=1000,  # Smaller for testing
        attention_heads=4   # Smaller for testing
    )
    
    print(f"\n🔍 Model parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")
    print(f"🔍 Non-trainable buffers: {sum(p.numel() for p in tokenizer.buffers()):,}")
    
    # Test forward pass
    print("\n🚀 Testing forward pass...")
    
    try:
        with torch.no_grad():
            output = tokenizer(frequency_tokens, demographic_info)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: ({batch_size}, {num_bands}, {output_features})")
        
        # Verify output shape
        assert output.shape == (batch_size, num_bands, output_features), \
            f"Shape mismatch: expected ({batch_size}, {num_bands}, {output_features}), got {output.shape}"
        
        print("✅ Output shape validation passed!")
        
        # Check for NaN/Inf values
        assert not torch.isnan(output).any(), "NaN values detected in output"
        assert not torch.isinf(output).any(), "Inf values detected in output"
        
        print("✅ No NaN/Inf values detected!")
        
        # Test value ranges
        output_min, output_max = output.min().item(), output.max().item()
        print(f"📈 Output value range: [{output_min:.3f}, {output_max:.3f}]")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test gradient flow
    print("\n🔄 Testing gradient flow...")
    
    try:
        # Enable gradients
        frequency_tokens.requires_grad_(True)
        output = tokenizer(frequency_tokens, demographic_info)
        
        # Compute dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert frequency_tokens.grad is not None, "No gradients computed for input"
        assert not torch.isnan(frequency_tokens.grad).any(), "NaN gradients detected"
        
        print("✅ Gradient flow test passed!")
        
        # Check trainable parameters have gradients
        param_count = 0
        grad_count = 0
        for name, param in tokenizer.named_parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None:
                    grad_count += 1
        
        print(f"📊 Trainable parameters: {param_count}, with gradients: {grad_count}")
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test attention weights extraction
    print("\n👁️ Testing attention weights extraction...")
    
    try:
        with torch.no_grad():
            attention_weights = tokenizer.get_attention_weights(frequency_tokens)
        
        expected_attn_shape = (batch_size, num_bands, tokenizer.attention_heads, num_channels, num_channels)
        print(f"   Attention weights shape: {attention_weights.shape}")
        print(f"   Expected shape: {expected_attn_shape}")
        
        assert attention_weights.shape == expected_attn_shape, \
            f"Attention shape mismatch: expected {expected_attn_shape}, got {attention_weights.shape}"
        
        print("✅ Attention weights extraction passed!")
        
    except Exception as e:
        print(f"❌ Attention weights test failed: {e}")
        return False
    
    # Test GPU compatibility (if available)
    if torch.cuda.is_available():
        print("\n🔥 Testing GPU compatibility...")
        
        try:
            # Move to GPU
            tokenizer_gpu = tokenizer.cuda()
            freq_tokens_gpu = frequency_tokens.cuda()
            demo_info_gpu = demographic_info.cuda()
            
            # Test forward pass on GPU
            with torch.no_grad():
                output_gpu = tokenizer_gpu(freq_tokens_gpu, demo_info_gpu)
            
            print(f"✅ GPU forward pass successful!")
            print(f"   GPU output shape: {output_gpu.shape}")
            print(f"   Device: {output_gpu.device}")
            
            # Verify kernels are on GPU
            assert tokenizer_gpu.rocket_kernels.device.type == 'cuda', "ROCKET kernels not on GPU"
            print("✅ ROCKET kernels successfully moved to GPU!")
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
    else:
        print("\n⚠️ GPU not available, skipping GPU tests")
    
    # Test with different batch sizes
    print("\n📊 Testing different batch sizes...")
    
    test_batch_sizes = [1, 2, 8, 16]
    for test_batch in test_batch_sizes:
        try:
            test_input = torch.randn(test_batch, num_bands, num_channels, time_points)
            test_demo = torch.randn(test_batch, 2)
            
            with torch.no_grad():
                test_output = tokenizer(test_input, test_demo)
            
            expected_shape = (test_batch, num_bands, output_features)
            assert test_output.shape == expected_shape, \
                f"Batch size {test_batch}: expected {expected_shape}, got {test_output.shape}"
            
            print(f"   ✅ Batch size {test_batch}: {test_output.shape}")
            
        except Exception as e:
            print(f"   ❌ Batch size {test_batch} failed: {e}")
            return False
    
    print("\n🎉 All MultiROCKET Tokenizer tests passed successfully!")
    print("\n📋 Summary:")
    print(f"   ✅ Forward pass with shape validation")
    print(f"   ✅ Gradient flow and backpropagation")
    print(f"   ✅ Attention weights extraction")
    print(f"   ✅ Multiple batch sizes")
    if torch.cuda.is_available():
        print(f"   ✅ GPU compatibility")
    print(f"   ✅ No NaN/Inf values")
    
    return True


if __name__ == "__main__":
    print("🚀 MultiROCKET Tokenizer Test Suite")
    print("=" * 50)
    
    success = test_multirocket_tokenizer()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("🔥 MultiROCKET Tokenizer is ready for integration!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")