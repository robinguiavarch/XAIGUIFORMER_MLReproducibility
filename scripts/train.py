"""
Training script for the XaiGuiFormer model on EEG connectome graphs.
Includes smart dataset splitting, training with gradient clipping,
balanced accuracy evaluation, and best model checkpointing.
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
import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Silence certain Captum warnings
warnings.filterwarnings("ignore", message="Setting forward, backward hooks and attributes on non-linear")

# Add src/ to Python path before any imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer


def evaluate_with_bac(model, loader, class_names=None, device="cpu"):
    """
    Evaluate the model using Balanced Accuracy and Accuracy metrics.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        loader (DataLoader): DataLoader for validation or test data.
        class_names (list, optional): List of class names. Not used directly.
        device (str or torch.device): Device to run evaluation on.

    Returns:
        dict: Dictionary containing coarse BAC, refined BAC, accuracy, and BAC gain.
    """
    model.eval()
    all_preds_coarse = []
    all_preds_refined = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_true = batch.y
            age = batch.age.view(-1, 1)
            gender = batch.gender.view(-1, 1)

            # Get frequency bounds
            if hasattr(batch, 'freq_bounds'):
                freq_bounds = batch.freq_bounds[0] if batch.freq_bounds.dim() > 1 else batch.freq_bounds
            else:
                freq_bounds = torch.tensor([[2., 4.], [4., 8.], [8., 10.], [10., 12.],
                                            [12., 18.], [18., 21.], [21., 30.],
                                            [30., 45.], [12., 30.]], device=device)

            try:
                logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
                all_preds_coarse.extend(logits_coarse.argmax(dim=1).cpu().tolist())
                all_preds_refined.extend(logits_refined.argmax(dim=1).cpu().tolist())
                all_targets.extend(y_true.cpu().tolist())
            except Exception:
                continue

    if not all_preds_refined:
        return {"bac_coarse": 0.0, "bac_refined": 0.0, "accuracy": 0.0, "bac_gain": 0.0}

    bac_coarse = balanced_accuracy_score(all_targets, all_preds_coarse)
    bac_refined = balanced_accuracy_score(all_targets, all_preds_refined)
    accuracy = accuracy_score(all_targets, all_preds_refined)

    return {
        "bac_coarse": float(bac_coarse),
        "bac_refined": float(bac_refined),
        "accuracy": float(accuracy),
        "bac_gain": float(bac_refined - bac_coarse)
    }


def load_graph_dataset(pickle_path):
    """
    Load a dataset of PyTorch Geometric graph objects from a pickle file.

    Args:
        pickle_path (str): Path to the pickle file.

    Returns:
        tuple: (list of graphs, bool indicating if stratified splitting is possible)
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
        class_counts = Counter(labels)
        return graphs, min(class_counts.values()) >= 2

    return graphs, False


def smart_split_dataset(graphs, test_size=0.2, val_size=0.1, random_state=42):
    """
    Perform adaptive train/val/test split, with logic for small datasets.

    Args:
        graphs (list): List of graph data objects.
        test_size (float): Proportion of test set.
        val_size (float): Proportion of validation set.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (train_graphs, val_graphs, test_graphs)
    """
    labels = [g.y.item() for g in graphs]
    class_counts = Counter(labels)
    min_class_size = min(class_counts.values())

    try:
        if len(graphs) < 10:
            if min_class_size >= 2:
                return train_test_split(graphs, test_size=test_size, random_state=random_state, stratify=labels), [], []
            return train_test_split(graphs, test_size=test_size, random_state=random_state), [], []
        elif len(graphs) < 30:
            train_graphs, temp = train_test_split(graphs, test_size=test_size + val_size,
                                                  random_state=random_state,
                                                  stratify=labels if min_class_size >= 2 else None)
            temp_labels = [g.y.item() for g in temp]
            val_graphs, test_graphs = train_test_split(temp, test_size=test_size / (test_size + val_size),
                                                       random_state=random_state,
                                                       stratify=temp_labels if min(Counter(temp_labels).values()) >= 2 else None)
            return train_graphs, val_graphs, test_graphs
        else:
            train_graphs, temp = train_test_split(graphs, test_size=0.3, random_state=random_state,
                                                  stratify=labels if min_class_size >= 2 else None)
            temp_labels = [g.y.item() for g in temp]
            val_graphs, test_graphs = train_test_split(temp, test_size=0.5, random_state=random_state,
                                                       stratify=temp_labels if min(Counter(temp_labels).values()) >= 2 else None)
            return train_graphs, val_graphs, test_graphs
    except ValueError:
        # Fallback on sequential slicing
        n_test = max(1, int(len(graphs) * test_size))
        n_val = max(1, int(len(graphs) * val_size))
        return graphs[n_test + n_val:], graphs[n_test:n_test + n_val], graphs[:n_test]


def train_epoch(model, loader, optimizer, device, epoch):
    """
    Perform one training epoch.

    Args:
        model (torch.nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer for backpropagation.
        device (torch.device): Device to use.
        epoch (int): Epoch number (for logging).

    Returns:
        float: Average training loss.
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

    return total_loss / max(1, num_batches)


if __name__ == "__main__":
    print("Starting XaiGuiFormer training")

    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/unified_connectome_graphs.pkl")
    print(f"Loading: {dataset_path}")
    try:
        all_graphs, can_stratify = load_graph_dataset(dataset_path)
        print(f"Loaded {len(all_graphs)} graphs")
    except Exception as e:
        print(f"Loading error: {e}")
        sys.exit(1)

    all_labels = [g.y.item() for g in all_graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)
    print(f"{num_classes} classes encoded")

    # Split
    train_graphs, val_graphs, test_graphs = smart_split_dataset(all_graphs)
    print(f"📊 Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # DataLoaders
    effective_batch_size = min(cfg.train.batch_size, len(train_graphs))
    print(f"Batch size: {effective_batch_size}")
    train_loader = DataLoader(train_graphs, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=effective_batch_size) if val_graphs else None
    test_loader = DataLoader(test_graphs, batch_size=effective_batch_size)

    # Model setup
    cfg.defrost()
    cfg.model.num_classes = num_classes
    cfg.freeze()
    model = XaiGuiFormer(cfg, training_graphs=train_graphs).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=cfg.train.optimizer.betas,
        eps=cfg.train.optimizer.eps,
        weight_decay=cfg.train.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    max_epochs = min(cfg.train.epochs, 100) if len(train_graphs) < 20 else cfg.train.epochs
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        print(f"\n{'='*50}\nEPOCH {epoch}/{max_epochs}\n{'='*50}")
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        if val_loader and epoch % 5 == 0:
            val_results = evaluate_with_bac(model, val_loader, label_encoder.classes_, device)
            print(f"Val BAC: {val_results['bac_refined']:.4f} (gain: {val_results['bac_gain']:+.4f})")

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
            print(f"Best model saved! Loss: {best_loss:.4f}")

        scheduler.step()

    # Final evaluation
    print(f"\n{'='*60}\nFINAL EVALUATION\n{'='*60}")
    if os.path.exists("checkpoints/xaiguiformer_best.pth"):
        checkpoint = torch.load("checkpoints/xaiguiformer_best.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded (epoch {checkpoint['epoch']})")

    final_results = evaluate_with_bac(model, test_loader, label_encoder.classes_, device)
    print(f"\nFINAL RESULTS:")
    print(f"   BAC Coarse:  {final_results['bac_coarse']:.4f}")
    print(f"   BAC Refined: {final_results['bac_refined']:.4f}")
    print(f"   XAI Gain:    {final_results['bac_gain']:+.4f}")
    print(f"   Accuracy:    {final_results['accuracy']:.4f}")

    torch.save(model.state_dict(), "checkpoints/xaiguiformer_final.pth")
    print(f"\nTraining completed! Best epoch: {best_epoch}")
