"""
Ablation Study Script – XAIguiFormer
Évalue les variantes du modèle avec ou sans certaines composantes clés (Explainer, dRoFE, Demographics).
"""

import os
import sys
import json
import torch
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader

# Chemin vers src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate

# =======================
# Paramètres
# =======================
SEEDS = [42, 123, 2024]
RESULTS_DIR = "results/ablation_study"
os.makedirs(RESULTS_DIR, exist_ok=True)

ABLATION_VARIANTS = {
    "FullModel": dict(use_xai_guidance=True, use_drofe=True, use_demographics=True),
    "w/o_Explainer": dict(use_xai_guidance=False, use_drofe=True, use_demographics=True),
    "w/o_dRoFE": dict(use_xai_guidance=True, use_drofe=False, use_demographics=True),
    "w/o_Demographics": dict(use_xai_guidance=True, use_drofe=True, use_demographics=False),
    "VanillaOnly": dict(use_xai_guidance=False, use_drofe=False, use_demographics=False),
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        freq_bounds = batch.freq_bounds
        y_true = batch.y

        if freq_bounds.dim() == 1:
            freq_bounds = freq_bounds.unsqueeze(0)

        optimizer.zero_grad()
        loss = model(batch, freq_bounds, age, gender, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/tdbrain_config.yaml")

    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/xai_graphs.pkl")
    all_graphs = load_dataset(dataset_path)
    all_labels = [g.y.item() for g in all_graphs]

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ablation_name, ablation_flags in ABLATION_VARIANTS.items():
        print(f"\nVariante: {ablation_name}")
        variant_results = []

        for seed in SEEDS:
            print(f"\n🔁 Seed = {seed}")
            set_seed(seed)

            train_graphs, test_graphs = train_test_split(
                all_graphs, test_size=0.2, random_state=seed, stratify=all_labels
            )

            train_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size, shuffle=True)
            test_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size)

            cfg.defrost()
            cfg.model.num_classes = num_classes
            cfg.model.use_xai_guidance = ablation_flags["use_xai_guidance"]
            cfg.model.use_drofe = ablation_flags["use_drofe"]
            cfg.model.use_demographics = ablation_flags["use_demographics"]
            cfg.freeze()

            model = XaiGuiFormer(cfg, training_graphs=train_graphs).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.train.optimizer.lr,
                betas=cfg.train.optimizer.betas,
                eps=cfg.train.optimizer.eps,
                weight_decay=cfg.train.optimizer.weight_decay
            )

            for epoch in range(1, cfg.train.epochs + 1):
                train_loss = train(model, train_loader, optimizer, device)
                print(f"[{ablation_name} | Seed {seed}] Epoch {epoch} | Loss = {train_loss:.4f}")

            # === Évaluation ===
            metrics = evaluate(model, test_loader, class_names=label_encoder.classes_, device=device)
            metrics["seed"] = seed
            variant_results.append(metrics)

            with open(os.path.join(RESULTS_DIR, f"{ablation_name}_seed{seed}.json"), "w") as f:
                json.dump(metrics, f, indent=4)

        # === Résumé par variante ===
        df = pd.DataFrame(variant_results)
        summary = {
            "mean": df.mean(numeric_only=True).round(4).to_dict(),
            "std": df.std(numeric_only=True).round(4).to_dict()
        }

        print(f"\nRésultats {ablation_name}")
        print(pd.DataFrame(summary))

        with open(os.path.join(RESULTS_DIR, f"{ablation_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
