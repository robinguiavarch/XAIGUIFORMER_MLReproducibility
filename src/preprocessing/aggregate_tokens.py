"""
Script pour agréger des graphes EEG individuels (1 par bande de fréquence)
en un unique objet PyG `Data` par sujet/session, contenant tous les
tokens (x_tokens), bornes de fréquences, et métadonnées nécessaires
pour le modèle XaiGuiFormer.

Entrée :
    - Fichier connectomes_graphs.pkl contenant une liste d'objets `Data` individuels.
      Chaque objet encode un graphe EEG construit à partir d’une bande et d’une méthode.

Sortie :
    - Fichier xai_graphs.pkl contenant un objet `Data` par sujet/session avec :
        - x_tokens: Tensor [Freq, d]
        - freq_bounds: Tensor [Freq, 2]
        - age, gender, y (label)
        - subject_id, session, methods, bands
"""

import os
import pickle
from collections import defaultdict
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_PATH = "data/TDBRAIN/tokens/connectomes_graphs.pkl"
OUTPUT_PATH = "data/TDBRAIN/tokens/xai_graphs.pkl"

def aggregate_graphs(graphs):
    """
    Agrège des graphes EEG individuels (par fréquence) en un seul graphe
    par sujet/session contenant tous les tokens nécessaires au modèle.

    Args:
        graphs (list[torch_geometric.data.Data]): Liste de graphes PyG,
            chacun correspondant à une bande de fréquence.

    Returns:
        list[torch_geometric.data.Data]: Liste de graphes agrégés, un par sujet/session.
            Chaque objet contient les attributs suivants :
            - x_tokens: Tensor [Freq, d]
            - freq_bounds: Tensor [Freq, 2]
            - y, age, gender
            - subject_id, session, methods, bands
    """
    grouped = defaultdict(list)

    # Grouper les graphes par (subject_id, session)
    for g in graphs:
        key = (g.subject_id, g.session)
        grouped[key].append(g)

    xai_graphs = []

    for (subject_id, session), graphs_list in tqdm(grouped.items(), desc="Agrégation"):
        x_tokens = []
        freq_bounds = []
        methods = []
        bands = []

        age = graphs_list[0].age
        gender = graphs_list[0].gender
        y = graphs_list[0].y

        for g in graphs_list:
            if hasattr(g, "x") and g.x.shape[0] > 1:
                # Convertit les poids en token vectorisé depuis la matrice de connectivité
                conn_size = g.x.shape[0]
                tril_len = (conn_size * (conn_size - 1)) // 2
                token = torch.zeros(tril_len)
                token[:min(tril_len, g.edge_attr.shape[0])] = g.edge_attr[:min(tril_len, g.edge_attr.shape[0])]
                x_tokens.append(token)
            else:
                print(f"Skip {subject_id} {session} : invalid x")
                continue

            freq_bounds.append(g.freq_bounds.squeeze())  # [2]
            methods.append(g.method)
            bands.append(g.band)

        if not x_tokens:
            continue

        data = Data(
            x_tokens=torch.stack(x_tokens),        # [Freq, d]
            freq_bounds=torch.stack(freq_bounds),  # [Freq, 2]
            y=y,
            age=age,
            gender=gender,
            subject_id=subject_id,
            session=session,
            methods=methods,
            bands=bands
        )
        xai_graphs.append(data)

    return xai_graphs


if __name__ == "__main__":
    print(f"Chargement des graphes depuis : {INPUT_PATH}")
    with open(INPUT_PATH, "rb") as f:
        graphs = pickle.load(f)
    print(graphs[0])

    print(f"{len(graphs)} graphes chargés. Agrégation en cours...")
    xai_graphs = aggregate_graphs(graphs)
    print(f"{len(xai_graphs)} graphes agrégés générés.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(xai_graphs, f)
    print(f"Fichier sauvegardé à : {OUTPUT_PATH}")

