"""
Training script for XaiGuiFormer model on EEG connectome graphs.
Corrected for concurrent XAI architecture compatibility.
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate, evaluate_with_bac


def load_graph_dataset(pickle_path):
    """Load PyG graph dataset from pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)
    
    # Validate required PyG attributes for XAI architecture
    required_attrs = ['x', 'edge_index', 'edge_attr', 'freqband_order', 'freq_bounds', 'age', 'gender', 'y']
    if graphs:
        sample = graphs[0]
        missing = [attr for attr in required_attrs if not hasattr(sample, attr)]
        if missing:
            raise ValueError(f"Missing graph attributes: {missing}")
        
        # Analyze class distribution
        labels = [g.y.item() for g in graphs]
        class_counts = Counter(labels)
        
        unique_classes = len(set(labels))
        min_class_size = min(class_counts.values())
        
        can_stratify = min_class_size >= 2
        return graphs, can_stratify
    
    return graphs, False


def smart_split_dataset(graphs, test_size=0.2, val_size=0.1, random_state=42):
    """Adaptive dataset splitting for small datasets."""
    labels = [g.y.item() for g in graphs]
    class_counts = Counter(labels)
    min_class_size = min(class_counts.values())
    
    if len(graphs) < 10:
        # Very small dataset: train/test only
        try:
            if min_class_size >= 2:
                train_graphs, test_graphs = train_test_split(
                    graphs, test_size=test_size, random_state=random_state, 
                    stratify=labels
                )
            else:
                train_graphs, test_graphs = train_test_split(
                    graphs, test_size=test_size, random_state=random_state
                )
            
            return train_graphs, [], test_graphs
            
        except ValueError:
            n_test = max(1, int(len(graphs) * test_size))
            test_graphs = graphs[:n_test]
            train_graphs = graphs[n_test:]
            return train_graphs, [], test_graphs
    
    elif len(graphs) < 30:
        # Small dataset: reduced validation
        try:
            if min_class_size >= 2:
                train_graphs, temp_graphs = train_test_split(
                    graphs, test_size=test_size + val_size, 
                    random_state=random_state, stratify=labels
                )
                
                temp_labels = [g.y.item() for g in temp_graphs]
                val_graphs, test_graphs = train_test_split(
                    temp_graphs, test_size=test_size/(test_size + val_size),
                    random_state=random_state, stratify=temp_labels
                )
            else:
                train_graphs, temp_graphs = train_test_split(
                    graphs, test_size=test_size + val_size, random_state=random_state
                )
                val_graphs, test_graphs = train_test_split(
                    temp_graphs, test_size=test_size/(test_size + val_size), 
                    random_state=random_state
                )
                
            return train_graphs, val_graphs, test_graphs
            
        except ValueError:
            n_test = max(1, int(len(graphs) * test_size))
            n_val = max(1, int(len(graphs) * val_size))
            
            test_graphs = graphs[:n_test]
            val_graphs = graphs[n_test:n_test + n_val]
            train_graphs = graphs[n_test + n_val:]
            
            return train_graphs, val_graphs, test_graphs
    
    else:
        # Normal dataset: standard split
        train_graphs, temp_graphs = train_test_split(
            graphs, test_size=0.3, random_state=random_state, 
            stratify=labels if min_class_size >= 2 else None
        )
        
        temp_labels = [g.y.item() for g in temp_graphs]
        val_graphs, test_graphs = train_test_split(
            temp_graphs, test_size=0.5, random_state=random_state,
            stratify=temp_labels if min(Counter(temp_labels).values()) >= 2 else None
        )
        
        return train_graphs, val_graphs, test_graphs


def train_epoch(model, loader, optimizer, device, epoch):
    """Training epoch with concurrent XAI processing."""
    model.train()
    total_loss = 0.
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        try:
            batch = batch.to(device)
            
            # Extract PyG batch components
            y_true = batch.y
            age = batch.age.view(-1, 1)
            gender = batch.gender.view(-1, 1)
            
            # Get frequency bounds (first sample, assuming consistent across batch)
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
            
        except Exception as e:
            continue

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    # Configuration
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/unified_connectome_graphs.pkl")
    
    try:
        all_graphs, can_stratify = load_graph_dataset(dataset_path)
    except Exception as e:
        sys.exit(1)

    # Label encoding
    all_labels = [g.y.item() for g in all_graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    # Dataset splitting
    train_graphs, val_graphs, test_graphs = smart_split_dataset(all_graphs)

    # DataLoaders
    effective_batch_size = min(cfg.train.batch_size, len(train_graphs))
    
    train_loader = DataLoader(train_graphs, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    
    if val_graphs:
        val_loader = DataLoader(val_graphs, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None
        
    test_loader = DataLoader(test_graphs, batch_size=effective_batch_size, shuffle=False, num_workers=0)

    # Model initialization
    cfg.defrost()
    cfg.model.num_classes = num_classes
    cfg.freeze()

    model = XaiGuiFormer(config=cfg, training_graphs=train_graphs).to(device)

    # Optimizer
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

    # Training loop
    max_epochs = min(cfg.train.epochs, 100) if len(train_graphs) < 20 else cfg.train.epochs
    
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validation
        if val_loader and epoch % 5 == 0:
            val_results = evaluate_with_bac(model, val_loader, label_encoder.classes_, device)
        
        # Save best model
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

    # Final evaluation
    if os.path.exists("checkpoints/xaiguiformer_best.pth"):
        checkpoint = torch.load("checkpoints/xaiguiformer_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = evaluate_with_bac(model, test_loader, label_encoder.classes_, device)
    
    torch.save(model.state_dict(), "checkpoints/xaiguiformer_final.pth")