"""
Script pour extraire des représentations vectorielles (tokens) à partir
des matrices de connectivité (e.g. wPLI, Coherence) du dataset TDBRAIN.

Chaque connectome est transformé en un vecteur en prenant la partie 
inférieure de la matrice (hors diagonale), et stocké avec ses métadonnées :
subject_id, session, méthode de connectivité et bande de fréquence.

Le résultat est sauvegardé dans un fichier .pkl pour une utilisation ultérieure
dans un pipeline de machine learning (GNN).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_lower_triangle(matrix):
    """
    Extrait la partie inférieure (en dessous de la diagonale) d’une matrice carrée,
    puis l’aplatit en un vecteur 1D.

    Args:
        matrix (np.ndarray): Matrice carrée de connectivité (NxN).

    Returns:
        np.ndarray: Vecteur 1D contenant les éléments de la partie inférieure.
    """
    return matrix[np.tril_indices_from(matrix, k=-1)]


def parse_filename(filename):
    """
    Extrait les métadonnées (session, méthode, bande) à partir d’un nom de fichier .npy.

    Args:
        filename (str): Nom de fichier au format 'ses-1_method_band.npy'.

    Returns:
        tuple: (session, method, band)
    """
    parts = filename.replace('.npy', '').split('_')
    return parts[0], parts[1], parts[2]

def build_token_dataset(connectome_root):
    """
    Construit un DataFrame de tokens à partir des fichiers .npy présents dans
    le dossier de connectomes. Chaque token représente un vecteur dérivé
    de la matrice de connectivité.

    Args:
        connectome_root (str or Path): Dossier contenant les sous-dossiers "sub-*".

    Returns:
        pd.DataFrame: Tableau contenant les colonnes :
            - subject_id
            - session
            - method
            - band
            - token (vecteur de connectivité)
    """
    data = []

    for subject_dir in sorted(Path(connectome_root).glob("sub-*")):
        subject_id = subject_dir.name
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            for npy_file in session_dir.glob("*.npy"):
                session, method, band = parse_filename(npy_file.name)
                matrix = np.load(npy_file)
                token = extract_lower_triangle(matrix)
                print(f"Dimension du token pour {subject_id}, {session}, {method}, {band}: {token.shape}")

                data.append({
                    "subject_id": subject_id,
                    "session": session,
                    "method": method,
                    "band": band,
                    "token": token
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Paramètres
    connectome_dir = "data/TDBRAIN/connectome"
    output_path = "data/TDBRAIN/tokens/connectome_tokens.pkl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Building token dataset...")
    df = build_token_dataset(connectome_dir)

    print(f"Saving tokens to: {output_path}")
    df.to_pickle(output_path)

    print(f"Done. {len(df)} tokens saved.")