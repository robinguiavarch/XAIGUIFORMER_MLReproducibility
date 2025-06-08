"""
Script d’entraînement pour le modèle complet XaiGuiFormer sur des graphes de connectomes EEG.
"""

import torch
from torch_geometric.data import DataLoader
import pickle
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Ajouter le chemin vers src pour importer config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.losses import compute_class_weights
from model.evaluator import evaluate


def load_graph_dataset(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        y_true = batch.y
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        freq_bounds = batch.freq_bounds[0]  # [Freq, 2]

        loss = model(batch, freq_bounds, age, gender, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    # === Charger la configuration
    cfg = get_cfg_defaults()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. Chargement des données ===
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "connectomes_graphs.pkl")
    all_graphs = load_graph_dataset(dataset_path)

    label_encoder = LabelEncoder()
    all_labels = [g.y.item() for g in all_graphs]
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers)

    # === 2. Initialisation du modèle ===
    cfg.model.num_classes = num_classes
    cfg.model.num_node_feat = train_graphs[0].x.shape[1]  # sécurité si non précisé

    model = XaiGuiFormer(config=cfg, training_graphs=train_graphs).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=cfg.train.optimizer.betas,
        eps=cfg.train.optimizer.eps,
        weight_decay=cfg.train.optimizer.weight_decay
    )

    # === 3. Entraînement ===
    all_metrics = []

    for epoch in range(1, cfg.train.epochs + 1):
        loss = train(model, train_loader, optimizer, device)

        metrics = evaluate(
            model=model,
            loader=test_loader,
            class_names=label_encoder.classes_,
            device=device,
            save_path=os.path.join(cfg.out_root, "results"),
            epoch=epoch
        )

        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

        all_metrics.append({
            "Epoch": epoch,
            "Loss": round(loss, 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "F1-macro": round(metrics["f1_macro"], 4),
            "F1-weighted": round(metrics["f1_weighted"], 4)
        })

    # === 4. Récapitulatif final ===
    df_metrics = pd.DataFrame(all_metrics)
    print("\nRésumé des performances :\n")
    print(df_metrics.to_string(index=False))
