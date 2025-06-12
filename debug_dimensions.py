import sys
import os
import pickle
import torch
from torch_geometric.loader import DataLoader

sys.path.append("src")
from config import get_cfg_defaults

def debug_dataset():
    cfg = get_cfg_defaults()
    
    # Charger les données
    dataset_path = "data/TDBRAIN/tokens/xai_graphs.pkl"
    with open(dataset_path, "rb") as f:
        graphs = pickle.load(f)
    
    print("=== DIAGNOSTIC COMPLET ===")
    print(f"Nombre de graphes: {len(graphs)}")
    
    if len(graphs) > 0:
        sample = graphs[0]
        print(f"x_tokens shape: {sample.x_tokens.shape}")
        print(f"freq_bounds shape: {sample.freq_bounds.shape}")
        print(f"age: {sample.age}")
        print(f"gender: {sample.gender}")
        print(f"y: {sample.y}")
        
        # Créer un mini-batch
        loader = DataLoader(graphs[:2], batch_size=2)
        batch = next(iter(loader))
        
        print("\n=== BATCH INFO ===")
        print(f"batch.x_tokens shape: {batch.x_tokens.shape}")
        print(f"batch.freq_bounds shape: {batch.freq_bounds.shape}")
        print(f"batch.age shape: {batch.age.shape}")
        print(f"batch.gender shape: {batch.gender.shape}")
        
        # Calculer la taille d'entrée pour token_projection
        token_size = sample.x_tokens.shape[-1]
        print(f"\n=== DIMENSIONS REQUISES ===")
        print(f"token_projection input size: {token_size}")
        
        # Estimer le nombre d'électrodes
        # Si tril_len = n*(n-1)/2, alors n ≈ sqrt(2*tril_len)
        import math
        estimated_electrodes = int(math.sqrt(2 * token_size + 0.25) + 0.5) + 1
        print(f"Nombre estimé d'électrodes: {estimated_electrodes}")
        
    print("==========================")

if __name__ == "__main__":
    debug_dataset()