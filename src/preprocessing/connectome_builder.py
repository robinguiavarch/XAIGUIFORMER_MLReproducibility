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
    ✅ VERSION CORRIGÉE : gère le fait que gender est déjà encodé en [1, 0]
    """
    df = pd.read_csv(tsv_path, sep="\t")
    
    # ✅ DIAGNOSTIC : afficher l'état initial
    print(f"=== DIAGNOSTIC LOAD_LABELS ===")
    print(f"Avant nettoyage - Total: {len(df)}")
    print(f"NaN dans indication: {df['indication'].isna().sum()}")
    print(f"NaN dans age: {df['age'].isna().sum()}")
    print(f"NaN dans gender: {df['gender'].isna().sum()}")
    print(f"Gender unique values: {df['gender'].unique()}")
    print(f"Gender dtype: {df['gender'].dtype}")
    
    # ✅ CORRECTION : nettoyer les NaN AVANT label encoding
    df = df.dropna(subset=["indication", "age", "gender"])
    print(f"Après nettoyage - Total: {len(df)}")
    
    # ✅ LABEL ENCODING pour indication
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["indication"])
    print(f"Labels créés: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # ✅ CORRECTION CRITIQUE : gender est déjà numérique [1, 0]
    # Pas besoin de mapping M/F → 1/0 !
    if df['gender'].dtype in ['int64', 'float64'] and set(df['gender'].unique()).issubset({0, 1, 0.0, 1.0}):
        print("✅ Gender déjà encodé en [0, 1] - conversion en float")
        df["gender"] = df["gender"].astype(float)
    else:
        print("✅ Gender en format texte - mapping M/F → 1/0")
        gender_mapping = {"M": 1.0, "F": 0.0, "m": 1.0, "f": 0.0}
        df["gender"] = df["gender"].map(gender_mapping)
    
    # ✅ VÉRIFICATION FINALE
    print(f"Après processing gender:")
    print(f"  - NaN count: {df['gender'].isna().sum()}")
    print(f"  - Unique values: {df['gender'].unique()}")
    print(f"  - Dtype: {df['gender'].dtype}")
    
    if df["gender"].isna().sum() > 0:
        print(f"❌ ATTENTION: {df['gender'].isna().sum()} NaN restants après gender processing!")
        df = df.dropna(subset=["gender"])
        print(f"Après suppression des NaN gender - Total: {len(df)}")

    # ✅ NORMALISER L'ÂGE (optionnel mais recommandé)
    age_min, age_max = df["age"].min(), df["age"].max()
    print(f"Âge - min: {age_min:.2f}, max: {age_max:.2f}")
    # df["age"] = (df["age"] - age_min) / (age_max - age_min)  # Normalisation [0,1]
    
    # Créer mapping pour chaque attribut
    label_map = dict(zip(df["participant_id"], df["label"]))
    age_map = dict(zip(df["participant_id"], df["age"]))
    gender_map = dict(zip(df["participant_id"], df["gender"]))
    
    print(f"✅ Mappings créés pour {len(label_map)} participants")
    print(f"Exemple mappings:")
    for i, pid in enumerate(list(label_map.keys())[:3]):
        print(f"  {pid}: label={label_map[pid]}, age={age_map[pid]:.2f}, gender={gender_map[pid]}")
    print(f"===========================")
    
    return label_map, age_map, gender_map, le


# ✅ Utiliser les mappings globaux (définis une seule fois)
participant_to_label, participant_to_age, participant_to_gender, label_encoder = load_labels()


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


def connectome_to_graph(connectome, subject_id, session, method, band, fmin, fmax, threshold=0.01):
    """
    ✅ VERSION CORRIGÉE : meilleure gestion des clés et debug
    """
    # ✅ EXTRACTION DE LA CLÉ PARTICIPANT
    subject_key = subject_id.split("_")[0]  # 'sub-88072845_ses-1' → 'sub-88072845'
    
    # ✅ RÉCUPÉRATION DES MÉTADONNÉES avec debug
    y_value = participant_to_label.get(subject_key, None)
    age_value = participant_to_age.get(subject_key, None)
    gender_value = participant_to_gender.get(subject_key, None)
    
    # ✅ DEBUG : afficher les valeurs récupérées
    if y_value is None or age_value is None or gender_value is None:
        print(f"❌ Missing data for {subject_key}:")
        print(f"   y_value: {y_value}")
        print(f"   age_value: {age_value}")
        print(f"   gender_value: {gender_value}")
        return None
    else:
        print(f"✅ {subject_key}: y={y_value}, age={age_value:.2f}, gender={gender_value}")

    # ✅ CONSTRUCTION DES EDGES
    edge_index = []
    edge_attr = []

    for i in range(connectome.shape[0]):
        for j in range(connectome.shape[1]):
            if i != j and connectome[i, j] > threshold:
                edge_index.append([i, j])
                edge_attr.append(connectome[i, j])

    if not edge_index:
        print(f"❌ No edges found for {subject_id} - threshold too high?")
        return None

    # ✅ CONVERSION EN TENSEURS
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.eye(connectome.shape[0], dtype=torch.float)
    y = torch.tensor([y_value], dtype=torch.long)

    # ✅ MÉTADONNÉES AVEC VÉRIFICATION
    freq_bounds = torch.tensor([[fmin, fmax]], dtype=torch.float)
    age = torch.tensor([age_value], dtype=torch.float)
    gender = torch.tensor([gender_value], dtype=torch.float)

    # ✅ VÉRIFICATION FINALE DES NaN
    if torch.isnan(age).any() or torch.isnan(gender).any():
        print(f"❌ NaN detected in tensors for {subject_id}!")
        print(f"   age tensor: {age}")
        print(f"   gender tensor: {gender}")
        return None

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        subject_id=subject_id,
        session=session,
        method=method,
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

    methods = cfg.connectome.methods
    freq_bands = {k: list(v) for k, v in cfg.connectome.frequency_band.items()}

    all_graphs = []
    
    print(f"=== DÉBUT BUILD_CONNECTOMES ===")
    print(f"Méthodes: {methods}")
    print(f"Bandes de fréquence: {list(freq_bands.keys())}")
    print(f"Participants disponibles: {len(participant_to_label)}")

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

                    save_dir = os.path.join(save_path, group, subj)
                    os.makedirs(save_dir, exist_ok=True)
                    out_file = os.path.join(save_dir, f"{subj}_{method}_{band}.npy")
                    np.save(out_file, con_matrix)

                    graph = connectome_to_graph(
                        con_matrix, subject_id, "ses-1", method, band, fmin, fmax, threshold=0.01
                    )
                    if graph:
                        all_graphs.append(graph)

    final_path = os.path.join(graph_save_path, "../tokens/connectomes_graphs.pkl")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(final_path, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"\n✅ All graphs saved to: {final_path} – total: {len(all_graphs)}")


if __name__ == "__main__":
    # ✅ CORRECTION : utiliser directement config.py sans fichier YAML externe
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    print("=== CONFIGURATION UTILISÉE ===")
    print(f"Dataset: {cfg.dataset}")
    print(f"Connectome methods: {cfg.connectome.methods}")
    print(f"Frequency bands: {list(cfg.connectome.frequency_band.keys())}")
    print(f"Data dir: {cfg.connectome.path.data_dir}")
    print(f"Save dir: {cfg.connectome.path.save_dir}")
    print("==============================")

    os.makedirs(cfg.connectome.path.save_dir, exist_ok=True)
    build_connectomes(cfg)