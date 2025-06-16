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
        if 'batch' in key:
            return 1 + getattr(self, key)[-1] if len(getattr(self, key)) > 0 else 0
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'frequency_tokens':
            return 0
        elif key == 'demographic_info':
            return 0
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class EEGFrequencyTokensDataset(InMemoryDataset):
    """
    Dataset for EEG frequency tokens data preprocessed with frequency band tokenization.
    """
    names = ['TDBRAIN']
    split_dataset = ['train', 'val', 'test']

    FREQUENCY_BANDS = {
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

    def __init__(self, root, name, split='train', transform=None, pre_transform=None, 
                 pre_filter=None, normalize=True):
        self.name = name
        assert self.name in self.names
        self.split = split
        assert self.split in self.split_dataset
        self.normalize = normalize
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[self.split_dataset.index(self.split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return os.path.join(self.root, f"{self.name}_reduced_timeseries", self.split)

    @property
    def processed_dir(self):
        return os.path.join(self.root, f"{self.name}_reduced_timeseries", "processed")

    @property
    def raw_file_names(self):
        return ['patients_metadata.json']

    @property
    def processed_file_names(self):
        return ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt']

    def download(self):
        pass

    def process(self):
        """
        Process the frequency tokens data into PyTorch tensors and create Data objects
        using uniform truncation to the minimum time length across all samples.
        """
        metadata_path = os.path.join(self.raw_dir, 'patients_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        expected_bands = 9
        expected_channels = 33

        all_tokens_data = []
        time_lengths = []

        for patient_id, patient_info in metadata.items():
            tokens_path = patient_info.get('frequency_tokens_path')
            if not tokens_path or not os.path.exists(tokens_path):
                continue

            try:
                frequency_tokens = np.load(tokens_path)
                if frequency_tokens.shape[0] != expected_bands:
                    continue
                if frequency_tokens.shape[1] != expected_channels:
                    continue
                all_tokens_data.append((patient_id, patient_info, frequency_tokens))
                time_lengths.append(frequency_tokens.shape[2])
            except Exception:
                continue

        if len(all_tokens_data) == 0:
            raise RuntimeError(f"No valid frequency tokens found in {self.raw_dir}")

        min_time_length = min(time_lengths)
        data_list = []

        for patient_id, patient_info, frequency_tokens in all_tokens_data:
            try:
                frequency_tokens = frequency_tokens[:, :, :min_time_length]
                if self.normalize:
                    for band_idx in range(frequency_tokens.shape[0]):
                        for ch_idx in range(frequency_tokens.shape[1]):
                            signal = frequency_tokens[band_idx, ch_idx, :]
                            mean = np.mean(signal)
                            std = np.std(signal)
                            if std > 0:
                                frequency_tokens[band_idx, ch_idx, :] = (signal - mean) / std
                frequency_tensor = torch.from_numpy(frequency_tokens).float()
                condition_code = patient_info.get('condition_code', 0)
                label = torch.tensor([condition_code]).float()
                age = patient_info.get('age', 0.0)
                gender = patient_info.get('gender', 0)
                demographic_info = torch.tensor([age, gender]).float()
                data = FrequencyTokensData(
                    frequency_tokens=frequency_tensor,
                    y=label,
                    demographic_info=demographic_info,
                    eid=patient_id
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            except Exception:
                continue

        if len(data_list) == 0:
            raise RuntimeError(f"No valid frequency tokens found in {self.raw_dir}.")

        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.split_dataset.index(self.split)])

    def __repr__(self):
        return f'{self.name}FrequencyTokensDataset({len(self)})'


class FrequencyTokensDataLoader:
    """
    DataLoader for EEG frequency tokens that returns batches in standard PyTorch format.
    """
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=0, min_length=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.min_length = min_length
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
        Custom collate function to create standardized batches with truncation.
        """
        tokens_list = [item.frequency_tokens for item in batch]
        y_list = [item.y for item in batch]
        demographic_info_list = [item.demographic_info for item in batch]
        eid_list = [item.eid for item in batch]

        time_lengths = [tokens.size(2) for tokens in tokens_list]
        min_length = self.min_length if self.min_length is not None else min(time_lengths)

        batch_size = len(batch)
        num_bands = tokens_list[0].size(0)
        num_channels = tokens_list[0].size(1)

        truncated_tokens = torch.zeros(batch_size, num_bands, num_channels, min_length)
        for i, tokens in enumerate(tokens_list):
            truncated_tokens[i] = tokens[:, :, :min_length]

        labels = torch.stack(y_list)
        demo_info = torch.stack(demographic_info_list)

        return {
            'frequency_tokens': truncated_tokens,
            'y': labels,
            'demographic_info': demo_info,
            'eid': eid_list
        }

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def get_frequency_bands_tensor():
    """
    Get the frequency bands tensor for model initialization.

    Returns:
        torch.Tensor: Tensor of shape [9, 2] representing [low_freq, high_freq] for each band.
    """
    bands = EEGFrequencyTokensDataset.FREQUENCY_BANDS
    freq_bands_list = [bands[name] for name in bands.keys()]
    return torch.tensor(freq_bands_list, dtype=torch.float32)