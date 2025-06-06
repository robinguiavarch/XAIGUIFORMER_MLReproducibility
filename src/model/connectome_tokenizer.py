import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_upper_triangle(matrix):
    # Renvoie la partie supérieure de la matrice (sans diagonale) aplatîe
    return matrix[np.triu_indices_from(matrix, k=1)]

def parse_filename(filename):
    # Extrait les métadonnées à partir du nom de fichier
    # Exemple : ses-1_wpli_theta.npy → ('ses-1', 'wpli', 'theta')
    parts = filename.replace('.npy', '').split('_')
    return parts[0], parts[1], parts[2]

def build_token_dataset(connectome_root):
    data = []

    for subject_dir in sorted(Path(connectome_root).glob("sub-*")):
        subject_id = subject_dir.name
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            for npy_file in session_dir.glob("*.npy"):
                session, method, band = parse_filename(npy_file.name)
                matrix = np.load(npy_file)
                token = extract_upper_triangle(matrix)
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
