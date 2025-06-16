import os
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# ==== MODULES IMPORTS ====
from utils.data_transformer_tensor_timeseries import EEGFrequencyTokensDataset, FrequencyTokensData
from models.xaiguiformer_timeseries import XAIguiFormerTimeSeries, get_frequency_bands_tensor
from utils.eval_metrics import eval_metrics

# ==== SEED FIX FOR REPRODUCIBILITY ====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======= CHUNKING FUNCTION =======
def create_chunks_from_dataset(dataset, chunk_size=2000, overlap=0.5):
    chunked_samples = []
    step_size = int(chunk_size * (1 - overlap))
    for sample in dataset:
        freq = sample.frequency_tokens
        tlen = freq.shape[2]
        for start in range(0, tlen - chunk_size + 1, step_size):
            chunked = freq[:, :, start:start + chunk_size]
            # On retourne des objets FrequencyTokensData !
            chunked_samples.append(
                FrequencyTokensData(
                    frequency_tokens=chunked,
                    y=sample.y,
                    demographic_info=sample.demographic_info,
                    eid=f"{sample.eid}_chunk_{start}"
                )
            )
    return chunked_samples

def custom_collate_fn(batch):
    # batch: list of FrequencyTokensData
    # on stacke chaque attribut individuellement (fréquence, labels, demo, eid)
    freq_tokens = torch.stack([x.frequency_tokens for x in batch])
    ys = torch.stack([x.y for x in batch])
    demo = torch.stack([x.demographic_info for x in batch])
    eids = [x.eid for x in batch]
    return {
        "frequency_tokens": freq_tokens,      # [batch, 9, 33, chunk]
        "y": ys,                              # [batch, 1]
        "demographic_info": demo,             # [batch, 2]
        "eid": eids
    }

def create_training_data(root, chunk_size=2000, overlap=0.5, batch_size=16):
    train_dataset = EEGFrequencyTokensDataset(root, "TDBRAIN", "train")
    val_dataset = EEGFrequencyTokensDataset(root, "TDBRAIN", "val")

    train_chunks = create_chunks_from_dataset(train_dataset, chunk_size, overlap)
    val_chunks = create_chunks_from_dataset(val_dataset, chunk_size, overlap)

    train_loader = DataLoader(
        train_chunks, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_chunks, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    return train_loader, val_loader

# ==== TRAINING AND EVAL LOOP ====
@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    total_loss, total_samples = 0, 0
    all_preds, all_labels = [], []
    criterion = torch.nn.CrossEntropyLoss()
    for batch in dataloader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        labels = batch['y'].squeeze().long()
        preds, _ = model(batch)
        loss = criterion(preds, labels)
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    avg_loss = total_loss / total_samples
    preds_logits = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels).long().view(-1)
    bac, aucpr, auroc = eval_metrics(preds_logits, labels_cat, num_classes=num_classes, device=device)
    return avg_loss, bac, aucpr, auroc

def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    project_root = Path(__file__).parent
    data_root = project_root / "data"
    batch_size = 16
    chunk_size = 2000
    overlap = 0.5
    epochs = 3
    num_classes = 4

    # Data loaders
    print("📦 Loading data & chunking ...")
    train_loader, val_loader = create_training_data(data_root, chunk_size, overlap, batch_size)
    print(f"✅ Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")

    # Model
    freq_bands = get_frequency_bands_tensor()
    model = XAIguiFormerTimeSeries(
        num_channels=33,
        num_classes=num_classes,
        freqband=freq_bands,
        num_kernels=200,
        attention_heads=2,
        output_features=128,
        num_heads=4,
        num_transformer_layers=12,
        mlp_ratio=4.0,
        init_values=0.001,
        dropout=0.1,
        attn_drop=0.0,
        droppath=0.0
    ).to(device)
    print(f"✅ Model ready: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    history = []

    best_val_bac, best_epoch = 0, 0
    for epoch in range(1, epochs + 1):
        print(f"\n===== EPOCH {epoch}/{epochs} =====")
        # ----- TRAIN -----
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            labels = batch['y'].squeeze().long()
            optimizer.zero_grad()
            preds, _ = model(batch)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if torch.isnan(v).any():
                        print(f"NaN détecté dans {k} (entrée)")
                    if torch.isinf(v).any():
                        print(f"Inf détecté dans {k} (entrée)")
            print("preds min/max:", preds.min().item(), preds.max().item())
            print("labels:", labels)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        train_loss = total_loss / total_samples

        # ----- EVAL -----
        val_loss, val_bac, val_aucpr, val_auroc = evaluate(model, val_loader, device, num_classes)
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val   loss: {val_loss:.4f} | BAC: {val_bac:.4f} | AUC-PR: {val_aucpr:.4f} | AUROC: {val_auroc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bac": val_bac,
            "val_aucpr": val_aucpr,
            "val_auroc": val_auroc
        })

        if val_bac > best_val_bac:
            best_val_bac, best_epoch = val_bac, epoch
            torch.save(model.state_dict(), "best_xaiguiformer_timeseries.pt")
            print("💾 Model saved (new best BAC!)")

    print(f"\nTraining finished! Best val BAC: {best_val_bac:.4f} at epoch {best_epoch}")

    # Tableau récapitulatif
    print("\n=== Résumé des métriques sur 3 epochs ===")
    print("{:<6} {:<12} {:<12} {:<10} {:<12} {:<12}".format("Epoch", "Train loss", "Val loss", "BAC", "AUC-PR", "AUROC"))
    for row in history:
        print("{:<6} {:<12.4f} {:<12.4f} {:<10.4f} {:<12.4f} {:<12.4f}".format(
            row['epoch'], row['train_loss'], row['val_loss'],
            row['val_bac'], row['val_aucpr'], row['val_auroc']
        ))

if __name__ == "__main__":
    main()
