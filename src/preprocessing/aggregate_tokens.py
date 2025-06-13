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
    Agrège des graphes EEG individuels (par bande de fréquence) en un seul graphe
    par sujet/session, contenant tous les tokens nécessaires à XaiGuiFormer.
    """
    grouped = defaultdict(list)
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
            if not hasattr(g, "x") or not hasattr(g, "edge_attr") or g.edge_attr is None:
                print(f"[WARN] Skip {subject_id}, {session}: graph missing x or edge_attr")
                continue

            if g.x.shape[0] < 2 or g.edge_attr.shape[0] < 1:
                print(f"[WARN] Skip {subject_id}, {session}: insufficient data")
                continue

            # Génération du token à partir de edge_attr
            conn_size = g.x.shape[0]
            tril_len = (conn_size * (conn_size - 1)) // 2
            token = torch.zeros(tril_len)
            token[:min(tril_len, g.edge_attr.shape[0])] = g.edge_attr[:min(tril_len, g.edge_attr.shape[0])]

            if token.sum().item() == 0.0:
                print(f"[WARN] Skip token with 0-valued edge_attr for {subject_id}, {session}")
                continue

            x_tokens.append(token)
            freq_bounds.append(g.freq_bounds.squeeze())
            methods.append(g.method)
            bands.append(g.band)

        if not x_tokens:
            print(f"[SKIP] {subject_id} {session} – aucun token valide")
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
    print(f"📥 Chargement des graphes depuis : {INPUT_PATH}")
    with open(INPUT_PATH, "rb") as f:
        graphs = pickle.load(f)

    print(f"✅ {len(graphs)} graphes unitaires chargés")
    xai_graphs = aggregate_graphs(graphs)
    print(f"✅ {len(xai_graphs)} graphes agrégés (prêts pour XaiGuiFormer)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(xai_graphs, f)

    print(f"💾 Fichier sauvegardé à : {OUTPUT_PATH}")
