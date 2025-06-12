"""
debug_train.py - Script de diagnostic pour identifier les problèmes dimensionnels
✅ CORRIGÉ : PYTHONPATH et imports
"""

import os
import sys
import pickle
import torch

# ✅ CORRECTION : PYTHONPATH depuis racine du projet  
sys.path.append("src")  # Pas "../src" !

# ✅ CORRECTION : Imports cohérents avec PYTHONPATH
from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from torch_geometric.loader import DataLoader


def load_and_inspect_data():
    """Charge et inspecte les données"""
    print("🔍 DIAGNOSTIC DES DONNÉES")
    print("=" * 50)
    
    cfg = get_cfg_defaults()
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/xai_graphs.pkl")
    
    with open(dataset_path, "rb") as f:
        graphs = pickle.load(f)
    
    print(f"📊 Dataset: {len(graphs)} graphes")
    
    # Inspecter un graphe sample
    sample = graphs[0]
    print(f"\n🔬 SAMPLE GRAPH INSPECTION:")
    print(f"   x_tokens shape: {sample.x_tokens.shape}")
    print(f"   x_tokens dtype: {sample.x_tokens.dtype}")
    print(f"   freq_bounds shape: {sample.freq_bounds.shape}")
    print(f"   age: {sample.age} (type: {type(sample.age)})")
    print(f"   gender: {sample.gender} (type: {type(sample.gender)})")
    print(f"   y: {sample.y} (type: {type(sample.y)})")
    
    # Inspecter les dimensions attendues
    print(f"\n📐 DIMENSIONS ATTENDUES:")
    print(f"   x_tokens: [Freq, d] = [18, 528]")
    print(f"   Après batching: [B*Freq, d] = [B*18, 528]")
    print(f"   Après reshape: [B, Freq, d] = [B, 18, 528]")
    
    return graphs


def test_dataloader(graphs):
    """Teste le DataLoader et le batching"""
    print(f"\n🔍 DIAGNOSTIC DATALOADER")
    print("=" * 50)
    
    # Prendre 3 échantillons pour test
    test_graphs = graphs[:3]
    loader = DataLoader(test_graphs, batch_size=3, shuffle=False)
    
    for batch_idx, batch in enumerate(loader):
        print(f"\n📦 BATCH {batch_idx}:")
        print(f"   batch.y: {batch.y}")
        print(f"   batch.y shape: {batch.y.shape}")
        print(f"   batch.x_tokens shape: {batch.x_tokens.shape}")
        print(f"   batch.freq_bounds shape: {batch.freq_bounds.shape}")
        print(f"   batch.age shape: {batch.age.shape}")
        print(f"   batch.gender shape: {batch.gender.shape}")
        
        # ✅ Test du reshape basé sur votre méthode
        B = batch.age.shape[0]  # Vraie batch size depuis age
        print(f"\n🔧 RESHAPE TEST (B={B}):")
        
        if batch.x_tokens.dim() == 2:
            total_tokens = batch.x_tokens.shape[0]
            d = batch.x_tokens.shape[1]
            Freq = total_tokens // B
            
            print(f"   total_tokens: {total_tokens}")
            print(f"   d: {d}")
            print(f"   Freq calculé: {Freq}")
            print(f"   B × Freq = {B} × {Freq} = {B * Freq}")
            print(f"   ✅ Cohérent: {total_tokens == B * Freq}")
            
            # Test reshape
            try:
                reshaped_tokens = batch.x_tokens.view(B, Freq, d)
                print(f"   ✅ Reshape réussi: {reshaped_tokens.shape}")
                
                # ✅ Test freq_bounds reshape aussi
                reshaped_freq = batch.freq_bounds[:Freq]  # Prendre les premiers Freq
                print(f"   ✅ freq_bounds shape: {reshaped_freq.shape}")
                
            except Exception as e:
                print(f"   ❌ Reshape échoué: {e}")
        
        break  # Un seul batch pour le test


def test_model_components(graphs):
    """Teste les composants du modèle individuellement"""
    print(f"\n🔍 DIAGNOSTIC MODÈLE")
    print("=" * 50)
    
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.model.num_classes = 9  # ✅ Adapter aux vraies données
    cfg.freeze()
    
    # Test du ConnectomeEncoder seul
    print(f"\n🧬 Test ConnectomeEncoder:")
    from model.gnn import ConnectomeEncoder
    
    # ✅ NOUVEAU : Test avec les vraies dimensions
    encoder = ConnectomeEncoder(
        node_in_features=cfg.model.num_node_feat,
        edge_in_features=cfg.model.num_edge_feat,
        node_hidden_features=cfg.model.dim_node_feat,
        edge_hidden_features=cfg.model.dim_edge_feat,
        output_features=cfg.model.dim_node_feat,
        num_gnn_layers=cfg.model.num_gnn_layer,
        dropout=cfg.model.dropout,
        gnn_type=cfg.model.gnn_type,
        num_freqband=18  # ✅ CORRIGÉ : vos données ont 18 freq bands
    )
    
    print(f"   Config input features: {cfg.model.num_node_feat}")
    print(f"   Config output features: {cfg.model.dim_node_feat}")
    print(f"   Réel x_tokens features: 528")
    
    # Test avec un échantillon
    sample = graphs[0]
    print(f"\n📊 Test forward ConnectomeEncoder:")
    print(f"   Input x_tokens: {sample.x_tokens.shape}")
    
    try:
        output = encoder(sample)
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   Expected: [1, 18, 128]")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        # ✅ DIAGNOSTIC : Regarder l'erreur spécifique
        print(f"\n🔍 DIAGNOSTIC ERREUR:")
        print(f"   L'erreur est probablement dans output_projection")
        print(f"   Il faut adapter : 528 → 128")


def test_simple_projection():
    """✅ NOUVEAU : Test projection simple 528→128"""
    print(f"\n🔍 TEST PROJECTION SIMPLE")
    print("=" * 50)
    
    from torch.nn import Linear
    
    # Test direct de la projection problématique
    projection = Linear(528, 128)
    test_input = torch.randn(1, 18, 528)  # Format après reshape
    
    print(f"   Input: {test_input.shape}")
    
    try:
        output = projection(test_input)
        print(f"   ✅ Output: {output.shape}")
        print(f"   ✅ Projection 528→128 fonctionne")
    except Exception as e:
        print(f"   ❌ Erreur projection: {e}")


def test_full_pipeline(graphs):
    """✅ NOUVEAU : Test du pipeline complet avec un échantillon"""
    print(f"\n🔍 TEST PIPELINE COMPLET")
    print("=" * 50)
    
    try:
        cfg = get_cfg_defaults()
        cfg.defrost()
        cfg.model.num_classes = 9
        cfg.freeze()
        
        # Créer le modèle
        model = XaiGuiFormer(config=cfg, training_graphs=graphs[:5])
        print(f"   ✅ Modèle créé")
        
        # Test avec un mini-batch
        test_loader = DataLoader(graphs[:2], batch_size=2, shuffle=False)
        batch = next(iter(test_loader))
        
        # Reshape comme dans votre logique
        B = batch.age.shape[0]
        total_tokens = batch.x_tokens.shape[0]
        Freq = total_tokens // B
        
        freq_bounds = batch.freq_bounds[:Freq]
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        
        print(f"   Input préparé: B={B}, Freq={Freq}")
        print(f"   freq_bounds: {freq_bounds.shape}")
        print(f"   age: {age.shape}")
        print(f"   gender: {gender.shape}")
        
        # Test forward sans target (pas de loss)
        logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
        
        print(f"   ✅ Forward réussi!")
        print(f"   logits_coarse: {logits_coarse.shape}")
        print(f"   logits_refined: {logits_refined.shape}")
        
    except Exception as e:
        print(f"   ❌ Erreur pipeline: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Diagnostic complet"""
    print("🚀 DIAGNOSTIC XAIGUIFORMER")
    print("=" * 60)
    
    try:
        # 1. Inspecter les données
        graphs = load_and_inspect_data()
        
        # 2. Tester le DataLoader
        test_dataloader(graphs)
        
        # 3. Tester projection simple
        test_simple_projection()
        
        # 4. Tester les composants du modèle
        test_model_components(graphs)
        
        # 5. ✅ NOUVEAU : Test pipeline complet
        test_full_pipeline(graphs)
        
        print(f"\n🎯 DIAGNOSTIC TERMINÉ")
        print("=" * 60)
        
        # 6. Recommandations finales
        print(f"\n💡 RECOMMANDATIONS:")
        print(f"   1. ✅ PYTHONPATH corrigé : sys.path.append('src')")
        print(f"   2. ✅ Imports corrigés : from config/model.* au lieu de src.*")
        print(f"   3. 🔧 Corriger ConnectomeEncoder projection : 528 → 128")
        print(f"   4. 🔧 Utiliser B = batch.age.shape[0] pour reshape")
        print(f"   5. 🔧 freq_bounds = batch.freq_bounds[:Freq]")
        
    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()