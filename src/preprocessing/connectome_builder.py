"""
Build connectivity matrices and convert them to PyG graph format for GNN processing.
Computes coherence and wPLI connectomes across frequency bands.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import mne
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mne_connectivity import spectral_connectivity_epochs
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from config import get_cfg_defaults


def load_subject_metadata(tsv_path="data/TDBRAIN/raw/participants.tsv"):
    """Load and process subject demographic and diagnostic information."""
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["indication", "age", "gender"])
    
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["indication"])

    if df["gender"].dtype == object:
        df["gender"] = df["gender"].map({"M": 1.0, "F": 0.0})

    label_map = dict(zip(df["participant_id"], df["label"]))
    age_map = dict(zip(df["participant_id"], df["age"]))
    gender_map = dict(zip(df["participant_id"], df["gender"]))
    
    return label_map, age_map, gender_map, le


def compute_connectivity_matrices(epochs, method, fmin, fmax, sfreq):
    """Compute connectivity matrix for given frequency band and method."""
    connectivity = spectral_connectivity_epochs(
        epochs, method=method, mode='fourier', sfreq=sfreq,
        fmin=fmin, fmax=fmax, faverage=True, verbose=False
    )
    return connectivity.get_data(output='dense')[:, :, 0]


def connectivity_matrix_to_graph(matrix, subject_id, method, band, fmin, fmax, 
                                metadata_maps, threshold=0.01):
    """
    Convert connectivity matrix to PyG graph format suitable for GNN.
    
    Args:
        matrix: Connectivity matrix [n_channels, n_channels]
        subject_id: Subject identifier
        method: Connectivity method (coh/wpli)
        band: Frequency band name
        fmin, fmax: Frequency bounds
        metadata_maps: Tuple of (label_map, age_map, gender_map, label_encoder)
        threshold: Edge threshold for graph construction
    """
    # ✅ FIXED: Handle 4 return values
    label_map, age_map, gender_map, _ = metadata_maps
    subject_key = subject_id.split("_")[0]
    
    # Get metadata
    y_value = label_map.get(subject_key)
    age_value = age_map.get(subject_key)
    gender_value = gender_map.get(subject_key)
    
    if any(v is None for v in [y_value, age_value, gender_value]):
        return None

    # Node features: connectivity strength (mean of each row)
    node_features = torch.tensor(
        np.mean(matrix, axis=1), dtype=torch.float
    ).unsqueeze(1)

    # Edge construction from connectivity matrix
    edge_indices = []
    edge_weights = []
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > threshold:
                edge_indices.append([i, j])
                edge_weights.append(matrix[i, j])

    if not edge_indices:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    # Graph metadata
    y = torch.tensor([y_value], dtype=torch.long)
    freq_bounds = torch.tensor([fmin, fmax], dtype=torch.float)
    age = torch.tensor([age_value], dtype=torch.float)
    gender = torch.tensor([gender_value], dtype=torch.float)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        subject_id=subject_id,
        method=method,
        band=band,
        freq_bounds=freq_bounds,
        age=age,
        gender=gender
    )


def process_subject_connectomes(epochs_file, subject_id, freq_bands, methods, 
                               metadata_maps, save_dir):
    """Process all frequency bands and methods for a single subject."""
    if not os.path.exists(epochs_file):
        return []

    epochs = mne.read_epochs(epochs_file, preload=True)
    if len(epochs) == 0:
        return []

    sfreq = epochs.info['sfreq']
    subject_graphs = []

    for method in methods:
        for band_name, (fmin, fmax) in freq_bands.items():
            # Compute connectivity
            connectivity_matrix = compute_connectivity_matrices(
                epochs, method, fmin, fmax, sfreq
            )
            
            if connectivity_matrix is None or connectivity_matrix.shape[0] == 0:
                continue

            # Save raw connectivity matrix
            matrix_save_dir = os.path.join(save_dir, subject_id.split("_")[0], "ses-1")
            os.makedirs(matrix_save_dir, exist_ok=True)
            matrix_file = os.path.join(matrix_save_dir, f"{subject_id}_{method}_{band_name}.npy")
            np.save(matrix_file, connectivity_matrix)

            # Convert to graph
            graph = connectivity_matrix_to_graph(
                connectivity_matrix, subject_id, method, band_name,
                fmin, fmax, metadata_maps
            )
            
            if graph is not None:
                subject_graphs.append(graph)

    return subject_graphs


def build_connectome_graphs(config):
    """Main function to build connectome graphs from preprocessed EEG data."""
    preprocessed_path = config.connectome.path.data_dir
    save_path = config.connectome.path.save_dir
    os.makedirs(save_path, exist_ok=True)

    # Load metadata - ✅ FIXED: Handle 4 return values
    metadata_maps = load_subject_metadata()
    
    # Configuration
    methods = config.connectome.methods
    freq_bands = {k: list(v) for k, v in config.connectome.frequency_band.items()}
    
    all_graphs = []

    # Process all subjects
    for group_dir in os.listdir(preprocessed_path):
        group_path = os.path.join(preprocessed_path, group_dir)
        if not os.path.isdir(group_path):
            continue

        for subject_dir in tqdm(os.listdir(group_path), desc=f"Processing {group_dir}"):
            subject_path = os.path.join(group_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue
                
            epochs_file = os.path.join(subject_path, f"{group_dir}_{subject_dir}_EC_epo.fif")
            subject_id = f"{group_dir}_{subject_dir}"
            
            subject_graphs = process_subject_connectomes(
                epochs_file, subject_id, freq_bands, methods,
                metadata_maps, save_path
            )
            
            all_graphs.extend(subject_graphs)

    # Save all graphs
    output_file = os.path.join(save_path, "../tokens/connectomes_graphs.pkl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "wb") as f:
        pickle.dump(all_graphs, f)


if __name__ == "__main__":
    config = get_cfg_defaults()
    config.freeze()
    
    build_connectome_graphs(config)