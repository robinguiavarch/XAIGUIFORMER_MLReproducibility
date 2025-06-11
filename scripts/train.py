"""
Script d’entraînement pour le modèle XaiGuiFormer sur des graphes de connectomes EEG
agrégés par sujet. Utilise un pipeline basé sur PyTorch Geometric avec évaluation
à chaque époque. Les graphes doivent contenir les attributs :
x_tokens, freq_bounds, age, gender, y.
"""

import os
import sys
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader  # nouvelle version recommandée

# Ajouter src/ au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate


def load_graph_dataset(pickle_path):
    """Charge le fichier .pkl contenant les graphes EEG"""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def train(model, loader, optimizer, device):
    """Boucle d'entraînement avec logging détaillé en cas d'erreur numérique"""
    model.train()
    total_loss = 0.

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        # === Préparation des inputs
        freq_bounds = batch.freq_bounds
        if freq_bounds.dim() == 1:
            freq_bounds = freq_bounds.unsqueeze(0)
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        y_true = batch.y

        optimizer.zero_grad()

        # === Forward
        try:
            loss = model(batch, freq_bounds, age, gender, y_true)

            if torch.isnan(loss):
                print("[Batch", i, "] Loss is NaN")
                print("y_true:", y_true)
                print("y_true dtype:", y_true.dtype)
                print("freq_bounds:", freq_bounds.shape)
                print("age:", age[:5])
                print("gender:", gender[:5])
                print("x_tokens mean:", batch.x_tokens.mean().item())
                print("x_tokens std:", batch.x_tokens.std().item())
                exit()

            # === Backward
            loss.backward()

            # === Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        except Exception as e:
            print(f"Exception in batch {i}: {e}")
            torch.save(batch, f"debug_batch_{i}.pt")
            raise e  # rethrow to debug

    return total_loss / len(loader)



if __name__ == "__main__":
    # === Charger la configuration
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/tdbrain_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. Chargement des graphes enrichis (xai_graphs.pkl) ===
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/xai_graphs.pkl")
    all_graphs = load_graph_dataset(dataset_path)

    label_encoder = LabelEncoder()
    all_labels = [g.y.item() for g in all_graphs]
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # === 2. Initialisation du modèle ===
    cfg.defrost()
    cfg.model.num_classes = num_classes
    cfg.freeze()

    model = XaiGuiFormer(config=cfg, training_graphs=train_graphs).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=cfg.train.optimizer.betas,
        eps=cfg.train.optimizer.eps,
        weight_decay=cfg.train.optimizer.weight_decay
    )

    # === 3. Boucle d'entraînement + évaluation ===
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

        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {metrics['accuracy']:.4f} | "
              f"F1-macro: {metrics['f1_macro']:.4f}")

        all_metrics.append({
            "Epoch": epoch,
            "Loss": round(loss, 4),
            "Accuracy": round(metrics["accuracy"], 4),
            "F1-macro": round(metrics["f1_macro"], 4),
            "F1-weighted": round(metrics["f1_weighted"], 4)
        })

    # === 4. Résumé final ===
    df_metrics = pd.DataFrame(all_metrics)
    print("\nRésumé des performances :\n")
    print(df_metrics.to_string(index=False))
