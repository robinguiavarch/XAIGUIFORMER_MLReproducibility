import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data, InMemoryDataset, Batch


# Add project root to Python path to enable imports from other modules
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


class FrequencyTokensData(Data):
    """
    A class to handle frequency tokens data for EEG time series.
    Adapted for frequency band tokens from MultiROCKET preprocessing.
    """
    def __inc__(self, key, value, *args, **kwargs):
        # Handle incrementing for batching
        if 'batch' in key:
            return 1 + getattr(self, key)[-1] if len(getattr(self, key)) > 0 else 0
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Define concatenation dimension for batching
        if key == 'frequency_tokens':
            return 0  # Concatenate along batch dimension for frequency tokens
        elif key == 'demographic_info':
            return 0  # Concatenate along batch dimension for demographic info
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class EEGFrequencyTokensDataset(InMemoryDataset):
    """
    Dataset for EEG frequency tokens data preprocessed with frequency band tokenization
    
    Args:
        root: Root directory where the dataset should be saved
        name: Name of the dataset (TDBRAIN)
        split: Dataset split ('train', 'val', or 'test')
        transform: A function/transform that takes in a Data object and returns a transformed version
        pre_transform: A function/transform that takes in a Data object and returns a transformed version
        pre_filter: A function that takes in a Data object and returns a boolean value
        normalize: Whether to normalize the frequency tokens (default: True)
    """
    names = ['TDBRAIN']
    split_dataset = ['train', 'val', 'test']
    
    # Frequency bands matching the original XAIguiFormer configuration
    FREQUENCY_BANDS = {
        'delta': [2., 4.],
        'theta': [4., 8.],
        'low_alpha': [8., 10.],
        'high_alpha': [10., 12.],
        'low_beta': [12., 18.],
        'mid_beta': [18., 21.],
        'high_beta': [21., 30.],
        'gamma': [30., 45.],
        'beta': [12., 30.]  # For theta/beta ratio
    }

    def __init__(self, root, name, split='train', transform=None, pre_transform=None, 
                 pre_filter=None, normalize=True):
        self.name = name
        assert self.name in self.names
        self.split = split
        assert self.split in self.split_dataset
        self.normalize = normalize
        
        print(f"🎵 Processing frequency tokens for {split} split...")
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed data
        path = self.processed_paths[self.split_dataset.index(self.split)]
        self.data, self.slices = torch.load(path)
        print(f"✅ Loaded {len(self)} frequency token samples from {split} set")

    @property
    def raw_dir(self):
        # Use the correct path structure: data/TDBRAIN_reduced_timeseries/split/
        return os.path.join(self.root, f"{self.name}_reduced_timeseries", self.split)

    @property
    def processed_dir(self):
        # Store processed data in: data/TDBRAIN_reduced_timeseries/processed/
        return os.path.join(self.root, f"{self.name}_reduced_timeseries", "processed")

    @property
    def raw_file_names(self):
        # Return metadata file which contains paths to all patient files
        return ['patients_metadata.json']

    @property
    def processed_file_names(self):
        return ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt']
    
    def download(self):
        # No download needed as data is already preprocessed
        pass

    def process(self):
        """
        Process the frequency tokens data into PyTorch tensors and create Data objects
        FIXED: Implements Option A - Truncation to minimum time length across all samples
        """
        # Load metadata file
        metadata_path = os.path.join(self.raw_dir, 'patients_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Expected dimensions
        expected_bands = 9
        expected_channels = 33
        
        print(f"📊 Processing {len(metadata)} patients...")
        
        # STEP 1: Load all tokens and find minimum time length
        print("🔍 Step 1: Loading all tokens and calculating minimum time length...")
        all_tokens_data = []
        time_lengths = []
        
        for patient_id, patient_info in metadata.items():
            # Get frequency tokens file path
            tokens_path = patient_info.get('frequency_tokens_path')
            if not tokens_path:
                print(f"⚠️  No frequency tokens path for {patient_id}")
                continue
                
            # Check if the file exists
            if not os.path.exists(tokens_path):
                print(f"⚠️  Tokens file not found: {tokens_path}")
                continue
            
            try:
                # Load frequency tokens (shape: [9, 33, time_points])
                frequency_tokens = np.load(tokens_path)
                
                # Validate dimensions
                if frequency_tokens.shape[0] != expected_bands:
                    print(f"⚠️  {patient_id}: {frequency_tokens.shape[0]} bands, expected {expected_bands}")
                    continue
                    
                if frequency_tokens.shape[1] != expected_channels:
                    print(f"⚠️  {patient_id}: {frequency_tokens.shape[1]} channels, expected {expected_channels}")
                    continue
                
                # Store data for processing
                all_tokens_data.append((patient_id, patient_info, frequency_tokens))
                time_lengths.append(frequency_tokens.shape[2])
                
                # Progress update
                if len(all_tokens_data) % 20 == 0:
                    print(f"   Loaded {len(all_tokens_data)} patients...")
                
            except Exception as e:
                print(f"❌ Error loading {patient_id}: {e}")
        
        if len(all_tokens_data) == 0:
            raise RuntimeError(f"No valid frequency tokens found in {self.raw_dir}")
        
        # Calculate statistics and minimum time length
        min_time_length = min(time_lengths)
        max_time_length = max(time_lengths)
        avg_time_length = np.mean(time_lengths)
        
        print(f"📏 Time length statistics:")
        print(f"   Min: {min_time_length} points ({min_time_length/90:.1f}s at 90Hz)")
        print(f"   Max: {max_time_length} points ({max_time_length/90:.1f}s at 90Hz)")
        print(f"   Avg: {avg_time_length:.1f} points ({avg_time_length/90:.1f}s at 90Hz)")
        print(f"   Difference: {max_time_length - min_time_length} points")
        print(f"🔧 Using minimum length {min_time_length} for uniform truncation")
        
        # STEP 2: Process all tokens with uniform truncation
        print("🔧 Step 2: Processing tokens with uniform truncation...")
        data_list = []
        
        for patient_id, patient_info, frequency_tokens in all_tokens_data:
            try:
                # Truncate to minimum time length BEFORE any other processing
                frequency_tokens = frequency_tokens[:, :, :min_time_length]
                
                # Log dimensions for the first file
                if len(data_list) == 0:
                    print(f"📈 First tokens shape after truncation: {frequency_tokens.shape}")
                
                # Normalize if requested
                if self.normalize:
                    # Normalize each frequency band and channel independently
                    # Shape: [bands, channels, time_points]
                    for band_idx in range(frequency_tokens.shape[0]):
                        for ch_idx in range(frequency_tokens.shape[1]):
                            signal = frequency_tokens[band_idx, ch_idx, :]
                            mean = np.mean(signal)
                            std = np.std(signal)
                            # Avoid division by zero
                            if std > 0:
                                frequency_tokens[band_idx, ch_idx, :] = (signal - mean) / std
                
                # Convert to PyTorch tensor
                frequency_tensor = torch.from_numpy(frequency_tokens).float()
                
                # Verify final tensor shape
                assert frequency_tensor.shape == (expected_bands, expected_channels, min_time_length), \
                    f"Unexpected final shape for {patient_id}: {frequency_tensor.shape}"
                
                # Get label (condition_code)
                condition_code = patient_info.get('condition_code', 0)
                label = torch.tensor([condition_code]).float()
                
                # Get demographic information (age and gender)
                age = patient_info.get('age', 0.0)
                gender = patient_info.get('gender', 0)
                demographic_info = torch.tensor([age, gender]).float()
                
                # Create FrequencyTokensData object
                data = FrequencyTokensData(
                    frequency_tokens=frequency_tensor,  # [9, 33, min_time_length] - ALL SAME SIZE
                    y=label,                            # [1]
                    demographic_info=demographic_info,  # [2]
                    eid=patient_id                      # Patient ID string
                )
                
                # Apply pre-filter if specified
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                
                # Apply pre-transform if specified
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                data_list.append(data)
                
                # Progress update
                if len(data_list) % 20 == 0:
                    print(f"   Processed {len(data_list)} patients...")
                
            except Exception as e:
                print(f"❌ Error processing {patient_id}: {e}")
                import traceback
                traceback.print_exc()
        
        if len(data_list) == 0:
            raise RuntimeError(f"No valid frequency tokens found in {self.raw_dir}. Check file paths and formats.")
        
        print(f"✅ Successfully processed {len(data_list)} patients with uniform shape")
        print(f"   Final tensor shape for all samples: [9, 33, {min_time_length}]")
        
        # Create directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Collate and save data - should work now with uniform shapes
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.split_dataset.index(self.split)])
        
        print(f"💾 Saved processed data to {self.processed_paths[self.split_dataset.index(self.split)]}")

    def __repr__(self):
        return f'{self.name}FrequencyTokensDataset({len(self)})'


class FrequencyTokensDataLoader:
    """
    DataLoader for EEG frequency tokens that returns batches in standard PyTorch format.
    
    Uses truncation instead of padding to handle variable length sequences.
    
    Args:
        dataset: EEGFrequencyTokensDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of subprocesses for data loading
        min_length: Optional minimum length to truncate all sequences (if None, uses min length in batch)
    """
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=0, min_length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.min_length = min_length
        
        # Use a standard PyTorch DataLoader with custom collate function
        from torch.utils.data import DataLoader
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.custom_collate
        )
    
    def custom_collate(self, batch):
        """
        Custom collate function to create standardized batches with truncation
        NOTE: Since dataset now has uniform shapes, this is simplified
        """
        # Extract data from batch
        tokens_list = [item.frequency_tokens for item in batch]
        y_list = [item.y for item in batch]
        demographic_info_list = [item.demographic_info for item in batch]
        eid_list = [item.eid for item in batch]
        
        # Since all tokens now have the same shape, we can stack directly
        # But keep truncation logic for safety/flexibility
        time_lengths = [tokens.size(2) for tokens in tokens_list]  # 3rd dimension is time
        
        # Use truncation if min_length is specified, otherwise all should be same size
        if self.min_length is not None:
            min_length = self.min_length
        else:
            min_length = min(time_lengths)  # Should be same for all now
        
        # Create truncated frequency tokens tensor
        batch_size = len(batch)
        num_bands = tokens_list[0].size(0)     # Should be 9
        num_channels = tokens_list[0].size(1)  # Should be 33
        
        truncated_tokens = torch.zeros(batch_size, num_bands, num_channels, min_length)
        
        # Fill with truncated data (should be no-op truncation now)
        for i, tokens in enumerate(tokens_list):
            truncated_tokens[i] = tokens[:, :, :min_length]  # Truncate time dimension
        
        # Stack labels and demographics
        labels = torch.stack(y_list)
        demo_info = torch.stack(demographic_info_list)
        
        return {
            'frequency_tokens': truncated_tokens,  # [batch, 9, 33, min_time_points]
            'y': labels,                           # [batch, 1]
            'demographic_info': demo_info,         # [batch, 2]
            'eid': eid_list                        # List[str]
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def test_frequency_tokens_dataloader():
    """
    Test function to verify the dimensions and functionality of the FrequencyTokensDataLoader
    """
    print("🧪 Testing FrequencyTokensDataLoader...")
    
    # Get project root path
    project_root = Path(__file__).parent.parent
    
    # Define paths using project root
    root = str(project_root / "data")
    name = "TDBRAIN"
    split = "train"
    
    print(f"🔍 Looking for data in: {os.path.join(root, name+'_reduced_timeseries', split)}")
    
    try:
        # Create dataset
        dataset = EEGFrequencyTokensDataset(root, name, split)
        print(f"✅ Successfully loaded {len(dataset)} samples from {split} set")
        
        # Create FrequencyTokensDataLoader
        batch_size = 3
        dataloader = FrequencyTokensDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            min_length=None  # Use min length in each batch
        )
        
        print(f"📊 DataLoader created with batch_size={batch_size}")
        
        # Get first batch
        for batch in dataloader:
            frequency_tokens = batch['frequency_tokens']
            labels = batch['y']
            demo_info = batch['demographic_info']
            eids = batch['eid']
            
            print("\n--- Frequency Tokens Batch Format ---")
            print(f"Frequency tokens shape: {frequency_tokens.shape} [batch, bands, channels, time]")
            print(f"Labels shape: {labels.shape} [batch, 1]")
            print(f"Demographic info shape: {demo_info.shape} [batch, 2]")
            print(f"Patient IDs: {eids}")
            
            # Check if shapes match expectations
            assert frequency_tokens.shape[0] == batch_size, f"Batch dimension mismatch: {frequency_tokens.shape[0]} != {batch_size}"
            assert frequency_tokens.shape[1] == 9, f"Frequency bands mismatch: {frequency_tokens.shape[1]} != 9"
            assert frequency_tokens.shape[2] == 33, f"Channels mismatch: {frequency_tokens.shape[2]} != 33"
            assert labels.shape[0] == batch_size, "Labels batch dimension mismatch"
            assert demo_info.shape[0] == batch_size, "Demographic info batch dimension mismatch"
            assert demo_info.shape[1] == 2, "Demographic info should have 2 features (age, gender)"
            
            print("✅ All dimension checks passed!")
            
            # Additional validation
            print("\n--- Data Validation ---")
            print(f"Frequency bands: {frequency_tokens.shape[1]}")
            print(f"Channels per band: {frequency_tokens.shape[2]}")
            print(f"Time points (uniform): {frequency_tokens.shape[3]}")
            print(f"Data type: {frequency_tokens.dtype}")
            print(f"Value range: [{frequency_tokens.min():.3f}, {frequency_tokens.max():.3f}]")
            
            # Check for NaN or Inf values
            assert not torch.isnan(frequency_tokens).any(), "NaN values detected in frequency tokens"
            assert not torch.isinf(frequency_tokens).any(), "Inf values detected in frequency tokens"
            
            print("✅ Data validation passed!")
            
            # Information for MultiROCKET integration
            print("\n--- MultiROCKET Integration Info ---")
            print(f"Input format for MultiROCKET: {frequency_tokens.shape}")
            print(f"Expected: [batch_size, 9_freq_bands, 33_channels, time_points]")
            print(f"Demographics: {demo_info.shape} (age, gender)")
            print(f"Number of classes: 4 (ADHD=0, MDD=1, SMC=2, HEALTHY=3)")
            print(f"Sampling rate: 90 Hz (from preprocessing)")
            
            # Test multiple batches for consistency
            print("\n--- Testing batch consistency ---")
            batch_shapes = []
            for i, test_batch in enumerate(dataloader):
                if i >= 2:  # Test 3 batches total
                    break
                batch_shapes.append(test_batch['frequency_tokens'].shape)
                print(f"Batch {i+2} shape: {test_batch['frequency_tokens'].shape}")
            
            # Check time dimension consistency (should be same now due to uniform truncation)
            time_dims = [shape[3] for shape in batch_shapes]
            print(f"Time dimensions across batches: {time_dims}")
            if len(set(time_dims)) == 1:
                print("✅ All batches have consistent time dimensions (as expected with uniform truncation)")
            else:
                print("⚠️  Time dimension variation detected - this shouldn't happen with the fixed implementation")
            
            return True, dataset, dataloader
            
            break  # Just test the first batch in detail
            
    except Exception as e:
        print(f"❌ Error testing FrequencyTokensDataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def get_frequency_bands_tensor():
    """
    Get the frequency bands tensor for model initialization (following original architecture)
    
    Returns:
        torch.Tensor: Frequency bands tensor [9, 2] with [low_freq, high_freq] for each band
    """
    bands = EEGFrequencyTokensDataset.FREQUENCY_BANDS
    freq_bands_list = [bands[name] for name in bands.keys()]
    return torch.tensor(freq_bands_list, dtype=torch.float32)


if __name__ == "__main__":
    print("🚀 Testing FIXED FrequencyTokensDataLoader implementation...")
    
    # Test the dataloader
    success, dataset, dataloader = test_frequency_tokens_dataloader()
    
    if success:
        print("\n🎉 FrequencyTokensDataLoader test completed successfully!")
        
        # Show how to get frequency bands for model
        freq_bands = get_frequency_bands_tensor()
        print(f"\n📊 Frequency bands tensor for model initialization:")
        print(f"Shape: {freq_bands.shape}")
        print(f"Content: {freq_bands}")
        
    else:
        print("\n❌ FrequencyTokensDataLoader test failed. Please check the error messages above.")