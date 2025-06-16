"""
Training script for the XaiGuiFormer model on EEG connectome graphs.
Supports concurrent XAI-guided architecture, adaptive data splitting,
balanced accuracy evaluation, and checkpointing.
"""

import os
import sys
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
from collections import Counter

# Add the src/ directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate, evaluate_with_bac


def load_graph_dataset(pickle_path):
    """
    Load a pickled PyTorch Geometric dataset of EEG connectome graphs.

    Args:
        pickle_path (str): Path to the dataset (.pkl file).

    Returns:
        Tuple[List[Data], bool]: List of graphs and a boolean indicating
                                 whether stratified splitting is possible.
    """
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)
    
    required_attrs = ['x', 'edge_index', 'edge_attr', 'freqband_order',
                      'freq_bounds', 'age', 'gender', 'y']
    
    if graphs:
        sample = graphs[0]
        missing = [attr for attr in required_attrs if not hasattr(sample, attr)]
        if missing:
            raise ValueError(f"Missing graph attributes: {missing}")
        
        labels = [g.y.item() for g in graphs]
        min_class_size = min(Counter(labels).values())
        can_stratify = min_class_size >= 2
        return graphs, can_stratify
    
    return graphs, False


def smart_split_dataset(graphs, test_size=0.2, val_size=0.1, random_state=42):
    """
    Perform adaptive train/val/test split, optimized for small datasets.

    Args:
        graphs (List[Data]): List of graph samples.
        test_size (float): Proportion of the dataset for testing.
        val_size (float): Proportion for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[List[Data], List[Data], List[Data]]: Split graphs for train, val, and test.
    """
    labels = [g.y.item() for g in graphs]
    min_class_size = min(Counter(labels).values())

    try:
        if len(graphs) < 10:
            if min_class_size >= 2:
                train, test = train_test_split(graphs, test_size=test_size,
                                               random_state=random_state, stratify=labels)
            else:
                train, test = train_test_split(graphs, test_size=test_size, random_state=random_state)
            return train, [], test

        elif len(graphs) < 30:
            train, temp = train_test_split(graphs, test_size=test_size + val_size,
                                           random_state=random_state,
                                           stratify=labels if min_class_size >= 2 else None)
            temp_labels = [g.y.item() for g in temp]
            val, test = train_test_split(temp,
                                         test_size=test_size / (test_size + val_size),
                                         random_state=random_state,
                                         stratify=temp_labels if min(Counter(temp_labels).values()) >= 2 else None)
            return train, val, test

        else:
            train, temp = train_test_split(graphs, test_size=0.3, random_state=random_state,
                                           stratify=labels if min_class_size >= 2 else None)
            temp_labels = [g.y.item() for g in temp]
            val, test = train_test_split(temp, test_size=0.5, random_state=random_state,
                                         stratify=temp_labels if min(Counter(temp_labels).values()) >= 2 else None)
            return train, val, test
    except ValueError:
        # Fallback: sequential slicing
        n_test = max(1, int(len(graphs) * test_size))
        n_val = max(1, int(len(graphs) * val_size))
        test = graphs[:n_test]
        val = graphs[n_test:n_test + n_val]
        train = graphs[n_test + n_val:]
        return train, val, test


def train_epoch(model, loader, optimizer, device, epoch):
    """
    Run one training epoch for the model.

    Args:
        model (nn.Module): XaiGuiFormer model.
        loader (DataLoader): DataLoader for training samples.
        optimizer (Optimizer): Optimizer used for training.
        device (torch.device): Device to perform computation.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        try:
            batch = batch.to(device)
            y_true = batch.y
            age = batch.age.view(-1, 1)
            gender = batch.gender.view(-1, 1)
            freq_bounds = batch.freq_bounds[0] if batch.freq_bounds.dim() > 1 else batch.freq_bounds

            optimizer.zero_grad()
            loss = model(batch, freq_bounds, age, gender, y_true)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        except Exception:
            continue

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    # === Configuration ===
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset Loading ===
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/unified_connectome_graphs.pkl")
    try:
        all_graphs, can_stratify = load_graph_dataset(dataset_path)
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        sys.exit(1)

    # === Label Encoding ===
    all_labels = [g.y.item() for g in all_graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    # === Dataset Splitting ===
    train_graphs, val_graphs, test_graphs = smart_split_dataset(all_graphs)

    # === DataLoaders ===
    batch_size = min(cfg.train.batch_size, len(train_graphs))
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size) if val_graphs else None
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # === Model Initialization ===
    cfg.defrost()
    cfg.model.num_classes = num_classes
    cfg.freeze()
    model = XaiGuiFormer(cfg, training_graphs=train_graphs).to(device)

    # === Optimizer & Scheduler ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=cfg.train.optimizer.betas,
        eps=cfg.train.optimizer.eps,
        weight_decay=cfg.train.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=1e-6
    )

    # === Training Loop ===
    max_epochs = min(cfg.train.epochs, 100) if len(train_graphs) < 20 else cfg.train.epochs
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        if val_loader and epoch % 5 == 0:
            val_metrics = evaluate_with_bac(model, val_loader, label_encoder.classes_, device)
            print(f"[Epoch {epoch}] Val BAC: {val_metrics['bac_refined']:.4f} (Δ: {val_metrics['bac_gain']:+.4f})")

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'config': cfg
            }, "checkpoints/xaiguiformer_best.pth")

        scheduler.step()

    # === Final Evaluation ===
    if os.path.exists("checkpoints/xaiguiformer_best.pth"):
        checkpoint = torch.load("checkpoints/xaiguiformer_best.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    final_metrics = evaluate_with_bac(model, test_loader, label_encoder.classes_, device)
    torch.save(model.state_dict(), "checkpoints/xaiguiformer_final.pth")

    print("\nFinal Test Results:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
