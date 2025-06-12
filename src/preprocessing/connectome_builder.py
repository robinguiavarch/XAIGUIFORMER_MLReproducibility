import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
import mne
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from mne_connectivity import spectral_connectivity_epochs
from mne import make_fixed_length_epochs
from torch_geometric.data import Data

# Ajouter src/ au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from config import get_cfg_defaults

warnings.filterwarnings("ignore")


def load_labels(tsv_path="data/TDBRAIN/raw/participants.tsv"):
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["indication", "age", "gender"])
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["indication"])

    # Normaliser le genre (ex: M/F en 0/1)
        # Ne pas mapper à nouveau si la colonne est déjà numérique
    if df["gender"].dtype == object:
        df["gender"] = df["gender"].map({"M": 1.0, "F": 0.0})

    # Créer mapping pour chaque attribut
    label_map = dict(zip(df["participant_id"], df["label"]))
    age_map = dict(zip(df["participant_id"], df["age"]))
    gender_map = dict(zip(df["participant_id"], df["gender"]))
    return label_map, age_map, gender_map, le


participant_to_label, participant_to_age, participant_to_gender, label_encoder = load_labels()

def compute_dual_connectomes(epochs, fmin, fmax, sfreq):
    coh = spectral_connectivity_epochs(
        epochs, method="coh", mode='fourier', sfreq=sfreq,
        fmin=fmin, fmax=fmax, faverage=True, verbose=False)
    wpli = spectral_connectivity_epochs(
        epochs, method="wpli", mode='fourier', sfreq=sfreq,
        fmin=fmin, fmax=fmax, faverage=True, verbose=False)

    coh_matrix = coh.get_data(output='dense')[:, :, 0]
    wpli_matrix = wpli.get_data(output='dense')[:, :, 0]
    return coh_matrix, wpli_matrix


def compute_connectivity(epochs, method, fmin, fmax, sfreq):
    con = spectral_connectivity_epochs(
        epochs,
        method=method,
        mode='fourier',
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False,
    )
    return con.get_data(output='dense')[:, :, 0]


def connectome_to_graph(coh_matrix, wpli_matrix, subject_id, session, band, fmin, fmax, threshold=0.01):
    subject_key = subject_id.split("_")[0]
    y_value = participant_to_label.get(subject_key, None)
    age_value = participant_to_age.get(subject_key, None)
    gender_value = participant_to_gender.get(subject_key, None)

    if y_value is None or age_value is None or gender_value is None:
        return None

    # node features: mean of rows (simplified approach)
    x = torch.tensor(np.mean(coh_matrix, axis=1), dtype=torch.float).unsqueeze(1)

    edge_index = []
    edge_attr = []

    for i in range(wpli_matrix.shape[0]):
        for j in range(wpli_matrix.shape[1]):
            if i != j and wpli_matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(wpli_matrix[i, j])

    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([y_value], dtype=torch.long)
    freq_bounds = torch.tensor([[fmin, fmax]], dtype=torch.float)
    age = torch.tensor([age_value], dtype=torch.float)
    gender = torch.tensor([gender_value], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        subject_id=subject_id,
        session=session,
        method="COH+WPLI",
        band=band,
        freq_bounds=freq_bounds,
        age=age,
        gender=gender
    )



def build_connectomes(cfg):
    preprocessed_path = cfg.connectome.path.data_dir
    save_path = cfg.connectome.path.save_dir
    graph_save_path = save_path
    os.makedirs(graph_save_path, exist_ok=True)

    freq_bands = {k: list(v) for k, v in cfg.connectome.frequency_band.items()}
    all_graphs = []

    for group in os.listdir(preprocessed_path):
        group_path = os.path.join(preprocessed_path, group)
        if not os.path.isdir(group_path):
            continue

        for subj in tqdm(os.listdir(group_path), desc=f"Processing {group}"):
            subj_path = os.path.join(group_path, subj)
            epoch_file = os.path.join(subj_path, f"{group}_{subj}_EC_epo.fif")

            if not os.path.exists(epoch_file):
                continue

            subject_id = f"{group}_{subj}"
            epochs = mne.read_epochs(epoch_file, preload=True)
            if len(epochs) == 0:
                continue

            # Découpe en 10 epochs de 3 secondes (si 30s d'enregistrement)
            sfreq = epochs.info['sfreq']
            epochs_3s = epochs

            for band, f_range in freq_bands.items():
                fmin, fmax = f_range

                # Calcul des connectomes cohérence + wPLI
                coh_matrix, wpli_matrix = compute_dual_connectomes(epochs_3s, fmin, fmax, sfreq)

                if coh_matrix is None or coh_matrix.shape[0] == 0:
                    continue

                # Sauvegarde brute des matrices
                save_dir = os.path.join(save_path, group, subj)
                os.makedirs(save_dir, exist_ok=True)
                out_file = os.path.join(save_dir, f"{subj}_COH+WPLI_{band}.npy")
                np.save(out_file, np.stack([coh_matrix, wpli_matrix]))

                # Construction du graphe PyG
                graph = connectome_to_graph(
                    coh_matrix, wpli_matrix, subject_id, "ses-1", band, fmin, fmax, threshold=0.01
                )
                if graph:
                    all_graphs.append(graph)

    final_path = os.path.join(graph_save_path, "../tokens/connectomes_graphs.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(all_graphs, f)

    print(f"\nAll graphs saved to: {final_path} – total: {len(all_graphs)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build connectomes and GNN graphs from EEG data")
    parser.add_argument("--config", type=str, default="configs/tdbrain_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.makedirs(cfg.connectome.path.save_dir, exist_ok=True)
    build_connectomes(cfg)
