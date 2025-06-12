import os
import sys
import json
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)
from torch_geometric.loader import DataLoader

# === Imports internes
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer


def load_graph_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model(
            batch,
            batch.freq_bounds.unsqueeze(0) if batch.freq_bounds.dim() == 1 else batch.freq_bounds,
            batch.age.view(-1, 1),
            batch.gender.view(-1, 1),
            batch.y
        )
        if torch.isnan(loss):
            print("Loss is NaN. Aborting...")
            exit()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        y_true = batch.y
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        freq_bounds = batch.freq_bounds
        if freq_bounds.dim() == 1:
            freq_bounds = freq_bounds.unsqueeze(0)

        _, logits = model(batch, freq_bounds, age, gender, y_true=None)
        probs = torch.softmax(logits, dim=1)

        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_targets.extend(y_true.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    bac = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    try:
        auroc = roc_auc_score(y_true, y_score, multi_class="ovr")
        auc_pr = average_precision_score(y_true, y_score, average="macro")
    except:
        auroc = auc_pr = float("nan")

    return {
        "BAC": bac,
        "AUROC": auroc,
        "AUC-PR": auc_pr,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


if __name__ == "__main__":
    output_dir = "results/eval_multi_seed"
    checkpoint_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs/tdbrain_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/xai_graphs.pkl")
    full_graphs = load_graph_dataset(dataset_path)

    all_labels = [g.y.item() for g in full_graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    cfg.model.num_classes = len(label_encoder.classes_)
    cfg.model.num_node_feat = full_graphs[0].x.shape[1]

    for seed in range(5):
        print(f"\nSeed {seed}")
        train_graphs, test_graphs = train_test_split(
            full_graphs, test_size=0.2, random_state=seed, stratify=all_labels
        )

        train_loader = DataLoader(train_graphs, batch_size=cfg.train.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size)

        model = XaiGuiFormer(cfg, training_graphs=train_graphs).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optimizer.lr,
            betas=cfg.train.optimizer.betas,
            eps=cfg.train.optimizer.eps,
            weight_decay=cfg.train.optimizer.weight_decay
        )

        best_bac = 0
        best_metrics = {}
        writer = SummaryWriter(log_dir=f"runs/tdbrain_seed_{seed}")
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = train(model, train_loader, optimizer, device)
            metrics = evaluate(model, test_loader, device)
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Test/Accuracy", metrics["accuracy"], epoch)
            writer.add_scalar("Test/BAC", metrics["BAC"], epoch)
            writer.add_scalar("Test/F1_macro", metrics["f1_macro"], epoch)

            if metrics["BAC"] > best_bac:
                best_bac = metrics["BAC"]
                best_metrics = metrics
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"tdbrain_seed_{seed}_best.pth"))
            writer.flush()
        writer.close()


        print(f"Best results (Seed {seed}):", best_metrics)
        all_results.append(best_metrics)

        with open(os.path.join(output_dir, f"metrics_seed_{seed}.json"), "w") as f:
            json.dump(best_metrics, f, indent=4)

    # Résumé
    df = pd.DataFrame(all_results)
    means = df.mean()
    stds = df.std()

    summary = {
        "mean": means.to_dict(),
        "std": stds.to_dict()
    }

    print("\nRésumé global (Moyenne ± Écart-type) :")
    for key in means.keys():
        print(f"{key}: {means[key]:.4f} ± {stds[key]:.4f}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
