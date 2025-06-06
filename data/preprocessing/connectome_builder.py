"""
Script pour générer les connectomes EEG à partir des données prétraitées
(en format .fif d’époques MNE). Il utilise la fonction `spectral_connectivity_epochs`
de MNE pour calculer les matrices de connectivité spectrale (e.g. wPLI, coherence)
dans différentes bandes de fréquence.

Chaque connectome est :
- sauvegardé sous forme de fichier .npy (matrice carrée)
- converti en graphe (torch_geometric.data.Data) et ajouté à une liste

La liste complète des graphes est sauvegardée dans un fichier .pkl.
Les paramètres (chemins, méthodes, bandes) sont définis dans un fichier de configuration YAML.
"""

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
from torch_geometric.data import Data

# Ajouter src/ au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from config import get_cfg_defaults

warnings.filterwarnings("ignore")


def load_labels(tsv_path="data/TDBRAIN/raw/participants.tsv"):
    """
    Charge les indications cliniques depuis le fichier participants.tsv
    et les encode en entiers pour l'apprentissage supervisé.

    Returns:
        participant_to_label (dict): Mapping ID sujet → label entier
        label_encoder (LabelEncoder): Pour décoder si besoin plus tard
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["indication"])
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["indication"])
    return dict(zip(df["participant_id"], df["label"])), le


participant_to_label, label_encoder = load_labels()


def compute_connectivity(epochs, method, fmin, fmax, sfreq):
    """
    Calcule la connectivité spectrale pour une plage de fréquence donnée
    et une méthode spécifique (wpli, coherence).

    Args:
        epochs (mne.Epochs): Époques EEG MNE.
        method (str): Méthode de connectivité ('wpli', 'coh').
        fmin (float): Fréquence minimale de la bande.
        fmax (float): Fréquence maximale de la bande.
        sfreq (float): Fréquence d’échantillonnage du signal EEG.

    Returns:
        np.ndarray: Matrice de connectivité de taille (n_channels, n_channels).
    """
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


def connectome_to_graph(connectome, subject_id, session, method, band, threshold=0.01):
    """
    Convertit une matrice de connectivité EEG (N x N) en objet torch_geometric.data.Data.

    Cette fonction :
    - Supprime les connexions faibles (inférieures à un seuil donné).
    - Encode les identifiants du sujet pour retrouver l'indication clinique (label).
    - Construit les arêtes du graphe (edge_index), les poids (edge_attr) et les features (x).
    - Associe un label encodé (y) pour l’apprentissage supervisé.

    Args:
        connectome (np.ndarray): Matrice carrée de connectivité EEG.
        subject_id (str): Identifiant du sujet (ex: 'sub-001').
        session (str): Nom de la session (ex: 'ses-1').
        method (str): Méthode de connectivité (ex: 'wpli').
        band (str): Bande de fréquence (ex: 'theta').
        threshold (float): Seuil minimal pour conserver une connexion.

    Returns:
        Data or None: Objet `torch_geometric.data.Data` contenant :
            - x : Matrice identité (features des nœuds)
            - edge_index : Indices des arêtes conservées (Tensor[2, E])
            - edge_attr : Poids des arêtes (Tensor[E])
            - y : Label du graphe (Tensor[1])
            - Autres attributs (subject_id, session, method, band)

        Retourne None si aucune connexion n’est au-dessus du seuil ou si le label est manquant.
    """
    subject_key = subject_id.split("_")[0]
    print(f"SUBJECT KEY !!!!!!!!  {subject_key}") # remove
    y_value = participant_to_label.get(subject_key, None)

    edge_index = []
    edge_attr = []

    for i in range(connectome.shape[0]):
        for j in range(connectome.shape[1]):
            if i != j and connectome[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(connectome[i, j])

    if not edge_index or y_value is None:
        print(f"Skipped {subject_id} ({method}/{band}) - nb_edges={len(edge_index)}, y_value={y_value}") # remove
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.eye(connectome.shape[0], dtype=torch.float)
    y = torch.tensor([y_value], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        subject_id=subject_id,
        session=session,
        method=method,
        band=band
    )


def build_connectomes(cfg):
    """
    Construit les connectomes EEG à partir des fichiers .fif
    et sauvegarde aussi les graphes correspondants en .pkl.

    Args:
        cfg (CfgNode): Configuration YAML.
    """
    preprocessed_path = cfg.connectome.path.data_dir
    save_path = cfg.connectome.path.save_dir
    graph_save_path = save_path
    os.makedirs(graph_save_path, exist_ok=True)

    methods = cfg.connectome.methods
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

            sfreq = epochs.info['sfreq']

            for method in methods:
                for band, f_range in freq_bands.items():
                    fmin, fmax = f_range
                    con_matrix = compute_connectivity(epochs, method, fmin, fmax, sfreq)

                    if con_matrix is None or con_matrix.shape[0] == 0:
                        continue

                    # Save matrix .npy
                    save_dir = os.path.join(save_path, group, subj)
                    os.makedirs(save_dir, exist_ok=True)
                    out_file = os.path.join(save_dir, f"{subj}_{method}_{band}.npy")
                    np.save(out_file, con_matrix)

                    # Convert to GNN graph
                    graph = connectome_to_graph(con_matrix, subject_id, "ses-1", method, band, threshold=0.01)
                    if graph: # remove
                        all_graphs.append(graph) # remove
                        print(f"Graph added: {subject_id} ({method}/{band}) -> label: {graph.y.item()}") # remove
                    else: # remove
                        print(f"Skipped {subject_id} ({method}/{band}) - no edges or missing label") # remove

    # Save all graphs
    final_path = os.path.join(graph_save_path, "connectomes_graphs.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"\nAll graphs saved to: {final_path} – total: {len(all_graphs)}") # remove


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build connectomes and GNN graphs from EEG data")
    parser.add_argument("--config", type=str, default="configs/tdbrain_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.makedirs(cfg.connectome.path.save_dir, exist_ok=True)
    build_connectomes(cfg)
