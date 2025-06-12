"""
Script multi-seed d’entraînement et d’évaluation du modèle XaiGuiFormer sur des connectomes EEG.
Évalue le modèle à chaque seed et agrège les métriques finales (Accuracy, F1-macro, F1-weighted).
"""

import os
import sys
import pickle
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

# Ajouter src/ au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate

SEEDS = [42, 123, 2024, 7, 99]
RESULTS_DIR = "results/eval_multi_seed"
os.makedirs(RESULTS_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_graph_dataset(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.

    for batch in loader:
        batch = batch.to(device)

        freq_bounds = batch.freq_bounds
        if freq_bounds.dim() == 1:
            freq_bounds = freq_bounds.unsqueeze(0)

        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        y_true = batch.y

        optimizer.zero_grad()

        loss = model(batch, freq_bounds, age, gender, y_true)

        if torch.isnan(loss):
            print("NaN loss détectée")
            exit()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/tdbrain_config.yaml")

    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/xai_graphs.pkl")
    all_graphs = load_graph_dataset(dataset_path)
    all_labels = [g.y.item() for g in all_graphs]

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows = []

    for seed in SEEDS:
        print(f"\nEntraînement avec seed = {seed}\n" + "-" * 50)

        set_seed(seed)

        train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=seed)

        train_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_workers)

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

        for epoch in range(1, cfg.train.epochs + 1):
            loss = train(model, train_loader, optimizer, device)

            # === Évaluation
            metrics = evaluate(
                model=model,
                loader=test_loader,
                class_names=label_encoder.classes_,
                device=device
            )
            print(
                f"[Seed {seed}] Epoch {epoch:02d} | "
                f"Loss: {loss:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1-macro: {metrics['f1_macro']:.4f} | "
                f"F1-weighted: {metrics['f1_weighted']:.4f}"
            )
            
        metrics["seed"] = seed
        print(f"\nRésultats finaux (seed={seed}):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")

        summary_rows.append({
            "Seed": seed,
            "Accuracy": metrics["accuracy"],
            "F1-macro": metrics["f1_macro"],
            "F1-weighted": metrics["f1_weighted"]
        })

        with open(os.path.join(RESULTS_DIR, f"metrics_seed_{seed}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    # === Agrégation finale
    df = pd.DataFrame(summary_rows)
    print("\nRésumé des performances multi-seed :")
    print(df.to_string(index=False))
    print("\nMoyenne :")
    print(df.mean(numeric_only=True).round(4))

    df.to_csv(os.path.join(RESULTS_DIR, "summary_all_seeds.csv"), index=False)
