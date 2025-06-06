"""
Script pour générer les connectomes EEG à partir des données prétraitées
(en format .fif d’époques MNE). Il utilise la fonction `spectral_connectivity_epochs`
de MNE pour calculer les matrices de connectivité spectrale (e.g. wPLI, coherence)
dans différentes bandes de fréquence.

Chaque connectome est sauvegardé sous forme de fichier .npy (matrice carrée).
Les paramètres (chemins, méthodes, bandes) sont définis dans un fichier de configuration YAML.
"""

import os
import mne
import numpy as np
import argparse
import sys
from tqdm import tqdm
import warnings
from mne_connectivity import spectral_connectivity_epochs

# Ajouter src/ au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from config import get_cfg_defaults

warnings.filterwarnings("ignore")


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


def build_connectomes(cfg):
    """
    Construit les connectomes EEG pour tous les sujets à partir des fichiers .fif
    Sauvegarde les matrices de connectivité dans des fichiers .npy.

    Args:
        cfg (CfgNode): Configuration chargée à partir d’un fichier YAML.
    """
    preprocessed_path = cfg.connectome.path.data_dir
    save_path = cfg.connectome.path.save_dir
    methods = cfg.connectome.methods
    freq_bands = {k: list(v) for k, v in cfg.connectome.frequency_band.items()}

    for group in os.listdir(preprocessed_path):
        group_path = os.path.join(preprocessed_path, group)
        if not os.path.isdir(group_path):
            continue

        for subj in tqdm(os.listdir(group_path), desc=f"Processing {group}"):
            subj_path = os.path.join(group_path, subj)
            epoch_file = os.path.join(subj_path, f"{group}_{subj}_EC_epo.fif")

            if not os.path.exists(epoch_file):
                continue

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build connectomes from preprocessed EEG data")
    parser.add_argument("--config", type=str, default="configs/tdbrain_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.makedirs(cfg.connectome.path.save_dir, exist_ok=True)
    build_connectomes(cfg)
