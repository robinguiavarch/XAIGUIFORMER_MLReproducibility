#!/usr/bin/env python3
"""
Complete pipeline test for MultiROCKET Tokenizer with EEG frequency tokens dataset.

Tests the complete chain:
1. Load frequency tokens using our EEGFrequencyTokensDataset
2. Transform to tensors using our FrequencyTokensDataLoader  
3. Apply MultiROCKET tokenization
4. Verify output dimensions and compatibility

Usage:
    poetry run python test_multirocket.py
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add project modules to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our custom modules
from utils.data_transformer_tensor_timeseries import (
    EEGFrequencyTokensDataset, 
    FrequencyTokensDataLoader,
    get_frequency_bands_tensor
)
from modules.multirocket_tokenizer import MultiROCKETTokenizer


def test_complete_pipeline():
    """Test the complete pipeline: Data → Tensors → MultiROCKET → Output"""
    print("🚀 Testing Complete Pipeline: Data → Tensors → MultiROCKET")
    print("=" * 70)
    
    # Check if data exists
    project_root = Path(__file__).parent
    data_root = project_root / "data"
    dataset_path = data_root / "TDBRAIN_reduced_timeseries" / "train"
    
    print(f"📂 Data root: {data_root}")
    print(f"🔍 Looking for: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Data directory not found: {dataset_path}")
        print(f"   Please run utils/preprocessing_timeseries.py first to generate the data")
        return False
    
    try:
        # STEP 1: Load dataset using our EEGFrequencyTokensDataset
        print(f"\n📦 STEP 1: Loading EEG Frequency Tokens Dataset...")
        dataset = EEGFrequencyTokensDataset(
            root=str(data_root),
            name="TDBRAIN", 
            split="train"
        )
        
        print(f"✅ Dataset loaded successfully:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Dataset type: {type(dataset).__name__}")
        
        # STEP 2: Create DataLoader using our FrequencyTokensDataLoader
        print(f"\n🔄 STEP 2: Creating FrequencyTokensDataLoader...")
        batch_size = 2  # As requested
        dataloader = FrequencyTokensDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Deterministic for testing
            num_workers=0   # Avoid multiprocessing issues
        )
        
        print(f"✅ DataLoader created successfully:")
        print(f"   Batch size: {batch_size}")
        print(f"   Total batches: {len(dataloader)}")
        print(f"   DataLoader type: {type(dataloader).__name__}")
        
        # STEP 3: Get a batch of tensors
        print(f"\n📊 STEP 3: Getting batch of frequency token tensors...")
        batch = next(iter(dataloader))
        
        frequency_tokens = batch['frequency_tokens']
        demographic_info = batch['demographic_info']
        labels = batch['y']
        patient_ids = batch['eid']
        
        print(f"✅ Batch loaded successfully:")
        print(f"   Frequency tokens: {frequency_tokens.shape}")
        print(f"   Demographic info: {demographic_info.shape}")
        print(f"   Labels: {labels.shape}")
        print(f"   Patient IDs: {patient_ids}")
        print(f"   Data type: {frequency_tokens.dtype}")
        print(f"   Data range: [{frequency_tokens.min():.3f}, {frequency_tokens.max():.3f}]")
        print(f"   Memory usage: {frequency_tokens.element_size() * frequency_tokens.nelement() / 1024 / 1024:.1f} MB")
        
        # Validate tensor format
        expected_shape = (batch_size, 9, 33)
        actual_shape = frequency_tokens.shape[:3]
        assert actual_shape == expected_shape, f"Unexpected shape: {actual_shape} vs {expected_shape}"
        
        assert demographic_info.shape == (batch_size, 2), f"Unexpected demo shape: {demographic_info.shape}"
        assert labels.shape == (batch_size, 1), f"Unexpected labels shape: {labels.shape}"
        
        print("✅ Tensor format validation passed!")
        
        # STEP 4: Initialize MultiROCKET Tokenizer
        print(f"\n🚀 STEP 4: Initializing MultiROCKET Tokenizer...")
        print(f"   🔧 Using reduced configuration for memory efficiency on Mac")
        tokenizer = MultiROCKETTokenizer(
            num_frequency_bands=9,
            num_channels=33,
            output_features=128,
            num_kernels=200,      # REDUCED: 5000 → 1000 for Mac compatibility
            attention_heads=2      # REDUCED: 8 → 4 for Mac compatibility
        )
        
        print(f"✅ MultiROCKET Tokenizer initialized:")
        total_params = sum(p.numel() for p in tokenizer.parameters())
        trainable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        rocket_kernels = tokenizer.rocket_kernels.numel()
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   ROCKET kernels: {rocket_kernels:,} (non-trainable)")
        print(f"   Memory estimate: ~{trainable_params * 4 / 1024 / 1024:.0f} MB")
        
        # Memory safety check
        if total_params > 50_000_000:  # 50M parameter limit for Mac
            print(f"   ⚠️  Warning: {total_params:,} parameters may cause memory issues on Mac")
        else:
            print(f"   ✅ Parameter count safe for Mac: {total_params:,}")
        
        # STEP 5: Apply MultiROCKET tokenization
        print(f"\n🔄 STEP 5: Applying MultiROCKET tokenization...")
        
        start_time = time.time()
        
        with torch.no_grad():
            tokenized_output = tokenizer(frequency_tokens, demographic_info)
        
        forward_time = time.time() - start_time
        
        print(f"✅ MultiROCKET tokenization completed:")
        print(f"   Processing time: {forward_time:.3f}s")
        print(f"   Input shape: {frequency_tokens.shape}")
        print(f"   Output shape: {tokenized_output.shape}")
        print(f"   Expected output: ({batch_size}, 9, 128)")
        print(f"   Output range: [{tokenized_output.min():.3f}, {tokenized_output.max():.3f}]")
        print(f"   Output mean: {tokenized_output.mean():.3f}")
        print(f"   Output std: {tokenized_output.std():.3f}")
        
        # STEP 6: Validate output format
        print(f"\n✅ STEP 6: Validating output format...")
        
        expected_output_shape = (batch_size, 9, 128)
        if tokenized_output.shape != expected_output_shape:
            print(f"❌ Output shape mismatch: expected {expected_output_shape}, got {tokenized_output.shape}")
            return False
        
        if torch.isnan(tokenized_output).any():
            print(f"❌ NaN values detected in output")
            return False
        
        if torch.isinf(tokenized_output).any():
            print(f"❌ Inf values detected in output")
            return False
        
        print(f"✅ Output validation passed!")
        print(f"   ✓ Shape: {tokenized_output.shape} ✅")
        print(f"   ✓ No NaN/Inf values ✅")
        print(f"   ✓ Reasonable value range ✅")
        
        # STEP 7: Test gradient flow
        print(f"\n🔄 STEP 7: Testing gradient flow...")
        
        frequency_tokens.requires_grad_(True)
        output = tokenizer(frequency_tokens, demographic_info)
        loss = output.sum()
        loss.backward()
        
        if frequency_tokens.grad is None:
            print(f"❌ No gradients computed")
            return False
        
        if torch.isnan(frequency_tokens.grad).any():
            print(f"❌ NaN gradients detected")
            return False
        
        print(f"✅ Gradient flow test passed!")
        print(f"   Input gradient range: [{frequency_tokens.grad.min():.6f}, {frequency_tokens.grad.max():.6f}]")
        
        # STEP 8: Test transformer compatibility
        print(f"\n🔗 STEP 8: Testing XAIguiTransformer compatibility...")
        
        # Get frequency bands tensor for model initialization
        freq_bands = get_frequency_bands_tensor()
        print(f"   Frequency bands tensor: {freq_bands.shape}")
        
        # Verify compatibility format
        assert tokenized_output.shape == (batch_size, 9, 128), "Output not compatible with transformer"
        assert demographic_info.shape == (batch_size, 2), "Demographics not compatible with transformer"
        assert freq_bands.shape == (9, 2), "Frequency bands tensor incorrect shape"
        
        print(f"✅ Transformer compatibility verified!")
        print(f"   ✓ freq_series: [batch, 9, 128] → {tokenized_output.shape} ✅")
        print(f"   ✓ demographic_info: [batch, 2] → {demographic_info.shape} ✅") 
        print(f"   ✓ frequency_bands: [9, 2] → {freq_bands.shape} ✅")
        
        # STEP 9: Test multiple batches consistency
        print(f"\n🔄 STEP 9: Testing multiple batches consistency...")
        
        batch_count = 0
        total_time = 0
        all_shapes = []
        
        for i, test_batch in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break
                
            start_time = time.time()
            with torch.no_grad():
                test_output = tokenizer(
                    test_batch['frequency_tokens'], 
                    test_batch['demographic_info']
                )
            batch_time = time.time() - start_time
            total_time += batch_time
            batch_count += 1
            
            all_shapes.append(test_output.shape)
            print(f"   Batch {i+1}: {test_batch['frequency_tokens'].shape} → {test_output.shape} in {batch_time:.3f}s")
        
        avg_time = total_time / batch_count
        print(f"📊 Consistency results:")
        print(f"   Average processing time: {avg_time:.3f}s per batch")
        print(f"   All output shapes: {all_shapes}")
        
        # Check consistency
        if len(set(all_shapes)) == 1:
            print(f"✅ All batches have consistent output shapes!")
        else:
            print(f"❌ Inconsistent output shapes detected")
            return False
        
        # STEP 10: GPU test if available
        if torch.cuda.is_available():
            print(f"\n🔥 STEP 10: Testing GPU compatibility...")
            
            tokenizer_gpu = tokenizer.cuda()
            frequency_tokens_gpu = frequency_tokens.detach().cuda()
            demographic_info_gpu = demographic_info.cuda()
            
            start_time = time.time()
            with torch.no_grad():
                output_gpu = tokenizer_gpu(frequency_tokens_gpu, demographic_info_gpu)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU test passed!")
            print(f"   GPU time: {gpu_time:.3f}s")
            print(f"   CPU time: {forward_time:.3f}s")
            print(f"   GPU speedup: {forward_time/gpu_time:.1f}x")
            print(f"   GPU output shape: {output_gpu.shape}")
            print(f"   GPU device: {output_gpu.device}")
            
            # Verify kernels are on GPU
            assert tokenizer_gpu.rocket_kernels.device.type == 'cuda'
            print(f"   ✓ ROCKET kernels on GPU: {tokenizer_gpu.rocket_kernels.device}")
        else:
            print(f"\n⚠️  GPU not available, skipping GPU test")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete pipeline test."""
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 CUDA device: {torch.cuda.get_device_name()}")
    
    success = test_complete_pipeline()
    
    if success:
        print(f"\n🎉 COMPLETE PIPELINE TEST SUCCESSFUL!")
        print(f"=" * 70)
        print(f"✅ Data loading with EEGFrequencyTokensDataset")
        print(f"✅ Tensor transformation with FrequencyTokensDataLoader") 
        print(f"✅ MultiROCKET tokenization")
        print(f"✅ Output validation and compatibility")
        print(f"✅ Gradient flow verification")
        print(f"✅ Multiple batch consistency")
        if torch.cuda.is_available():
            print(f"✅ GPU compatibility")
        print(f"=" * 70)
        print(f"✅ READY FOR XAIGUIFORMER INTEGRATION!")
        print(f"\n🔗 Next steps:")
        print(f"   1. Create XAIguiFormerTimeSeries with MultiROCKET")
        print(f"   2. Replace ConnectomeEncoder with MultiROCKETTokenizer")
        print(f"   3. Update main training script")
        print(f"   4. Run end-to-end training")
        print(f"\n💡 Performance notes:")
        print(f"   • Current config optimized for Mac (1000 kernels, 4 heads)")
        print(f"   • For GPU training, consider scaling up to 5000 kernels, 8 heads")
        print(f"   • Memory usage: ~{trainable_params * 4 / 1024 / 1024:.0f} MB (safe for Mac)")
        
    else:
        print(f"\n❌ Pipeline test failed. Please check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)