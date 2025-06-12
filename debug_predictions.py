import torch
import pickle
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append("src")

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer

# Charger données
with open("data/TDBRAIN/tokens/xai_graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

# Analyser distribution des classes
all_labels = [g.y.item() for g in graphs]
print("=== DISTRIBUTION DES CLASSES ===")
from collections import Counter
label_counts = Counter(all_labels)
print(f"Classes: {sorted(label_counts.keys())}")
print(f"Distribution: {label_counts}")
print(f"Total échantillons: {len(all_labels)}")

# Charger modèle entraîné
cfg = get_cfg_defaults()
cfg.defrost()
cfg.model.num_classes = len(set(all_labels))
cfg.freeze()

model = XaiGuiFormer(config=cfg, training_graphs=graphs)
model.load_state_dict(torch.load("checkpoints/xaiguiformer_final.pth", map_location="cpu"))
model.eval()

# Test sur quelques échantillons
loader = DataLoader(graphs[:4], batch_size=2, shuffle=False)
with torch.no_grad():
    for i, batch in enumerate(loader):
        # Reshape batching
        B = batch.y.shape[0]
        if batch.x_tokens.dim() == 2:
            Freq = batch.x_tokens.shape[0] // B
            d = batch.x_tokens.shape[1]
            batch.x_tokens = batch.x_tokens.view(B, Freq, d)
            batch.freq_bounds = batch.freq_bounds.view(B, Freq, 2)
        
        freq_bounds = batch.freq_bounds[0]
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        
        logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
        preds = logits_refined.argmax(dim=1)
        
        print(f"\n--- Batch {i} ---")
        print(f"Targets: {batch.y}")
        print(f"Logits refined: {logits_refined}")
        print(f"Predictions: {preds}")
        print(f"Max logit values: {logits_refined.max(dim=1)}")