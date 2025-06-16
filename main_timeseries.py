import os
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Import tes modules
from utils.data_transformer_tensor_timeseries import EEGFrequencyTokensDataset, FrequencyTokensData, FrequencyTokensDataLoader
from models.xaiguiformer_timeseries import XAIguiFormerTimeSeries, get_frequency_bands_tensor
from utils.eval_metrics import accuracy, f1_macro, get_classification_report  # adapte à tes besoins

# ===== SEED FIX (reproductibilité totale) =====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======= CHUNKING FONCTIONNEL =======

def create_chunks_from_dataset(dataset, chunk_size=2000, overlap=0.5, as_dict=True):
    chunked_samples = []
    step_size = int(chunk_size * (1 - overlap))
    for sample in dataset:
        freq = sample.frequency_tokens
        tlen = freq.shape[2]
        for start in range(0, tlen - chunk_size + 1, step_size):
            chunked = freq[:, :, start:start + chunk_size]
            if as_dict:
                chunked_samples.append({
                    "frequency_tokens": chunked,
                    "y": sample.y,
                    "demographic_info": sample.demographic_info,
                    "eid": f"{sample.eid}_chunk_{start}"
                })
            else:
                chunked_samples.append(
                    FrequencyTokensData(
                        frequency_tokens=chunked,
                        y=sample.y,
                        demographic_info=sample.demographic_info,
                        eid=f"{sample.eid}_chunk_{start}"
                    )
                )
    return chunked_samples

def create_training_data(root, chunk_size=2000, overlap=0.5, batch_size=16):
    train_dataset = EEGFrequencyTokensDataset(root, "TDBRAIN", "train")
    val_dataset = EEGFrequencyTokensDataset(root, "TDBRAIN", "val")

    train_chunks = create_chunks_from_dataset(train_dataset, chunk_size, overlap, as_dict=False)
    val_chunks = create_chunks_from_dataset(val_dataset, chunk_size, overlap, as_dict=False)

    train_loader = DataLoader(train_chunks, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_chunks, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

# ====== TRAINING LOOP ======
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for batch in dataloader:
        # batch: dict with tensors
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        labels = batch['y'].squeeze().long()
        optimizer.zero_grad()
        preds, _ = model(batch)  # [batch, num_classes]
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    for batch in dataloader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        labels = batch['y'].squeeze().long()
        preds, _ = model(batch)
        loss = criterion(preds, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.append(preds.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_macro(all_preds.numpy(), all_labels.numpy())
    return avg_loss, acc, f1

# ========= MAIN ==========
def main():
    # Device & config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    # Paths
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data"
    batch_size = 16
    chunk_size = 2000
    overlap = 0.5
    epochs = 100

    # Data loaders
    print("📦 Loading data & chunking ...")
    train_loader, val_loader = create_training_data(data_root, chunk_size, overlap, batch_size)
    print(f"✅ Training samples: {len(train_loader.dataset)} | Validation samples: {len(val_loader.dataset)}")

    # Model
    freq_bands = get_frequency_bands_tensor()
    model = XAIguiFormerTimeSeries(
        num_channels=33,
        num_classes=4,
        freqband=freq_bands,
        num_kernels=200,
        attention_heads=2,
        output_features=128,
        num_heads=4,
        num_transformer_layers=12,  # YAML
        mlp_ratio=4.0,
        init_values=0.001,
        dropout=0.1,
        attn_drop=0.0,
        droppath=0.0
    ).to(device)
    print(f"✅ Model ready: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")

    # Optimizer/loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc, best_epoch = 0, 0
    for epoch in range(1, epochs + 1):
        print(f"\n===== EPOCH {epoch}/{epochs} =====")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Train loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | acc: {val_acc:.4f} | f1_macro: {val_f1:.4f}")

        # Optionally save best
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), "best_xaiguiformer_timeseries.pt")
            print("💾 Model saved (new best!)")

    print(f"\nTraining finished! Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
