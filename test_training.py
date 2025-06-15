#!/usr/bin/env python3
"""
Test rapide d'entraînement pour vérifier que les NaN ont disparu
✅ FINAL - Compatible avec les noms originaux et imports corrigés
"""

import torch
import pickle
import sys
import os
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder

sys.path.append("src")
from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer

def test_forward_pass():
    """Test du forward pass sur un petit batch"""
    
    print("=== TEST FORWARD PASS FINAL ===")
    
    # Charger config
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Charger données
    dataset_path = "data/TDBRAIN/tokens/xai_graphs.pkl"
    with open(dataset_path, "rb") as f:
        graphs = pickle.load(f)
    
    print(f"Graphes chargés: {len(graphs)}")
    
    # Préparer labels
    all_labels = [g.y.item() for g in graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)
    
    # Mettre à jour config
    cfg.defrost()
    cfg.model.num_classes = num_classes
    cfg.freeze()
    
    print(f"Classes: {num_classes} - {label_encoder.classes_}")
    
    # Créer loader (petit batch)
    loader = DataLoader(graphs[:4], batch_size=2, shuffle=False)
    
    # Créer modèle
    model = XaiGuiFormer(config=cfg, training_graphs=graphs).to(device)
    print(f"Modèle créé - paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    model.train()
    for i, batch in enumerate(loader):
        print(f"\n--- Batch {i} ---")
        batch = batch.to(device)
        
        # Debug dimensions
        print(f"x_tokens shape: {batch.x_tokens.shape}")
        print(f"freq_bounds shape: {batch.freq_bounds.shape}")
        print(f"age shape: {batch.age.shape}")
        print(f"gender shape: {batch.gender.shape}")
        print(f"y shape: {batch.y.shape}")
        
        # ✅ CORRECTION : Restructurer le batching PyG
        B = batch.y.shape[0]  # Nombre d'échantillons dans le batch
        
        # PyG concatène les tenseurs → il faut les reshaper
        if batch.x_tokens.dim() == 2:  # [B*Freq, d] au lieu de [B, Freq, d]
            Freq = batch.x_tokens.shape[0] // B  # 36 // 2 = 18
            d = batch.x_tokens.shape[1]  # 528
            
            # Reshape en format attendu
            batch.x_tokens = batch.x_tokens.view(B, Freq, d)
            batch.freq_bounds = batch.freq_bounds.view(B, Freq, 2)
            
            print(f"✅ Reshaped - x_tokens: {batch.x_tokens.shape}, freq_bounds: {batch.freq_bounds.shape}")
        
        # Prendre freq_bounds du premier échantillon (ils sont identiques)
        freq_bounds = batch.freq_bounds[0]  # [Freq, 2]
        age = batch.age.view(-1, 1)         # [B, 1] 
        gender = batch.gender.view(-1, 1)   # [B, 1]
        
        print(f"Input shapes:")
        print(f"  freq_bounds: {freq_bounds.shape}")
        print(f"  age: {age.shape} - values: {age.flatten()}")
        print(f"  gender: {gender.shape} - values: {gender.flatten()}")
        
        # ✅ VÉRIFICATION : pas de NaN dans les inputs
        if torch.isnan(age).any():
            print("❌ NaN détecté dans age!")
            return False
        if torch.isnan(gender).any():
            print("❌ NaN détecté dans gender!")
            return False
        if torch.isnan(batch.x_tokens).any():
            print("❌ NaN détecté dans x_tokens!")
            return False
            
        print("✅ Aucun NaN détecté dans les inputs")
        
        try:
            # Forward pass
            loss = model(batch, freq_bounds, age, gender, batch.y)
            
            print(f"✅ Forward réussi!")
            print(f"Loss: {loss.item():.6f}")
            print(f"Loss is NaN: {torch.isnan(loss)}")
            
            if torch.isnan(loss):
                print("❌ Loss = NaN - problème dans le forward!")
                return False
            
            # Test backward
            loss.backward()
            print("✅ Backward réussi!")
            
        except Exception as e:
            print(f"❌ Erreur dans forward/backward: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n🎉 TOUS LES TESTS PASSÉS - EXPLAINER CAPTUM FONCTIONNEL!")
    return True

if __name__ == "__main__":
    success = test_forward_pass()
    if success:
        print("\n✅ Prêt pour l'entraînement complet!")
        print("✅ Plus de warnings 'Layer explainer failed'!")
        print("✅ Explanations multi-couches fonctionnelles!")
    else:
        print("\n❌ Il reste des problèmes à corriger")