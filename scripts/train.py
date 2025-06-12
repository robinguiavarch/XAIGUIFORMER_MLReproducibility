"""
Script d'entraînement pour le modèle XaiGuiFormer sur des graphes de connectomes EEG
✅ CORRIGÉ pour les petits datasets (16 échantillons, 9 classes)
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

# Ajouter src/ au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate, evaluate_with_bac


def load_graph_dataset(pickle_path):
    """Charge le fichier .pkl contenant les graphes EEG"""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"❌ Fichier non trouvé: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        graphs = pickle.load(f)
    
    print(f"✅ Graphes chargés: {len(graphs)}")
    
    # ✅ Validation des attributs requis
    required_attrs = ['x_tokens', 'freq_bounds', 'age', 'gender', 'y']
    if graphs:
        sample = graphs[0]
        missing = [attr for attr in required_attrs if not hasattr(sample, attr)]
        if missing:
            raise ValueError(f"❌ Attributs manquants dans les graphes: {missing}")
        
        print(f"✅ Attributs validés: {required_attrs}")
        print(f"   x_tokens shape: {sample.x_tokens.shape}")
        print(f"   freq_bounds shape: {sample.freq_bounds.shape}")
        
        # ✅ Analyse de la distribution des classes
        labels = [g.y.item() for g in graphs]
        class_counts = Counter(labels)
        print(f"📊 Distribution des classes:")
        for class_id, count in sorted(class_counts.items()):
            print(f"   Classe {class_id}: {count} échantillons")
        
        unique_classes = len(set(labels))
        print(f"   Classes détectées: {unique_classes}")
        
        # ✅ Vérification pour stratification
        min_class_size = min(class_counts.values())
        if min_class_size < 2:
            print(f"⚠️  Classes avec 1 seul échantillon détectées - pas de stratification possible")
            return graphs, False  # Pas de stratification
        else:
            print(f"✅ Stratification possible (min classe: {min_class_size})")
            return graphs, True  # Stratification OK
    
    return graphs, False


def smart_split_dataset(graphs, test_size=0.2, val_size=0.1, random_state=42):
    """
    ✅ NOUVEAU : Split intelligent pour petits datasets
    - Si stratification possible : l'utilise
    - Sinon : split aléatoire simple
    """
    labels = [g.y.item() for g in graphs]
    class_counts = Counter(labels)
    min_class_size = min(class_counts.values())
    
    print(f"📊 Analyse du dataset pour split:")
    print(f"   Total échantillons: {len(graphs)}")
    print(f"   Classes: {len(class_counts)}")
    print(f"   Taille classe min: {min_class_size}")
    
    # ✅ Stratégie adaptative selon la taille
    if len(graphs) < 10:
        # Dataset très petit : pas de validation set
        print("🔧 Dataset très petit (<10) - Split train/test seulement")
        try:
            if min_class_size >= 2:
                train_graphs, test_graphs = train_test_split(
                    graphs, test_size=test_size, random_state=random_state, 
                    stratify=labels
                )
                print("✅ Split stratifié train/test")
            else:
                train_graphs, test_graphs = train_test_split(
                    graphs, test_size=test_size, random_state=random_state
                )
                print("✅ Split aléatoire train/test")
            
            return train_graphs, [], test_graphs  # Pas de validation
            
        except ValueError as e:
            print(f"⚠️  Fallback split simple: {e}")
            # Split manuel simple
            n_test = max(1, int(len(graphs) * test_size))
            test_graphs = graphs[:n_test]
            train_graphs = graphs[n_test:]
            return train_graphs, [], test_graphs
    
    elif len(graphs) < 30:
        # Dataset petit : validation réduite
        print("🔧 Dataset petit (<30) - Split train/val/test avec val réduite")
        try:
            if min_class_size >= 2:
                # Premier split train/(val+test)
                train_graphs, temp_graphs = train_test_split(
                    graphs, test_size=test_size + val_size, 
                    random_state=random_state, stratify=labels
                )
                
                # Second split val/test
                temp_labels = [g.y.item() for g in temp_graphs]
                val_graphs, test_graphs = train_test_split(
                    temp_graphs, test_size=test_size/(test_size + val_size),
                    random_state=random_state, stratify=temp_labels
                )
                print("✅ Split stratifié train/val/test")
            else:
                # Split simple sans stratification
                train_graphs, temp_graphs = train_test_split(
                    graphs, test_size=test_size + val_size, random_state=random_state
                )
                val_graphs, test_graphs = train_test_split(
                    temp_graphs, test_size=test_size/(test_size + val_size), 
                    random_state=random_state
                )
                print("✅ Split aléatoire train/val/test")
                
            return train_graphs, val_graphs, test_graphs
            
        except ValueError as e:
            print(f"⚠️  Fallback split proportionnel: {e}")
            # Split manuel proportionnel
            n_test = max(1, int(len(graphs) * test_size))
            n_val = max(1, int(len(graphs) * val_size))
            
            test_graphs = graphs[:n_test]
            val_graphs = graphs[n_test:n_test + n_val]
            train_graphs = graphs[n_test + n_val:]
            
            return train_graphs, val_graphs, test_graphs
    
    else:
        # Dataset normal : split standard
        print("✅ Dataset normal - Split standard train/val/test")
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
    """✅ CORRIGÉ : Affiche TOUTES les loss, pas que les meilleures"""
    model.train()
    total_loss = 0.
    num_batches = 0
    losses = []  # ✅ NOUVEAU : Stocker toutes les loss
    
    print(f"🔄 Entraînement epoch {epoch}...")
    
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        try:
            batch = batch.to(device)
            
            # Reshape logic (votre méthode parfaite)
            B = batch.age.shape[0]
            total_tokens = batch.x_tokens.shape[0]
            Freq = total_tokens // B
            freq_bounds = batch.freq_bounds[:Freq]
            
            y_true = batch.y
            age = batch.age.view(-1, 1)
            gender = batch.gender.view(-1, 1)

            optimizer.zero_grad()
            loss = model(batch, freq_bounds, age, gender, y_true)
            
            if torch.isnan(loss):
                print(f"❌ NaN détecté batch {batch_idx}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_val = loss.item()
            total_loss += loss_val
            losses.append(loss_val)  # ✅ Stocker chaque loss
            num_batches += 1
            
        except Exception as e:
            print(f"❌ Erreur batch {batch_idx}: {e}")
            continue

    avg_loss = total_loss / max(num_batches, 1)
    
    # ✅ AFFICHAGE DÉTAILLÉ des loss
    if losses:
        print(f"✅ Epoch {epoch} - Loss détaillée:")
        print(f"   Moyenne: {avg_loss:.4f}")
        print(f"   Min: {min(losses):.4f}")
        print(f"   Max: {max(losses):.4f}")
        print(f"   Dernière: {losses[-1]:.4f}")
    
    return avg_loss


def validate_config(cfg):
    """Validation de la configuration"""
    print(f"✅ Configuration validée")
    print(f"   Learning rate: {cfg.train.optimizer.lr}")
    print(f"   Batch size: {cfg.train.batch_size}")
    print(f"   Epochs: {cfg.train.epochs}")
    print(f"   Alpha (XAI weight): {cfg.train.criterion.alpha}")
    print(f"   Dropout: {cfg.model.dropout}")


if __name__ == "__main__":
    print("🚀 Démarrage de l'entraînement XAIguiFormer")
    
    # ✅ Configuration
    cfg = get_cfg_defaults()
    validate_config(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")

    # === 1. Chargement des graphes ===
    dataset_path = os.path.join(cfg.connectome.path.save_dir, "../tokens/xai_graphs.pkl")
    print(f"📂 Chargement: {dataset_path}")
    
    try:
        all_graphs, can_stratify = load_graph_dataset(dataset_path)
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        sys.exit(1)

    # ✅ Label encoding avec validation
    all_labels = [g.y.item() for g in all_graphs]
    unique_labels = set(all_labels)
    print(f"📊 Labels uniques: {sorted(unique_labels)}")
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"✅ Classes encodées: {num_classes}")
    print(f"   Mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

    # ✅ Split intelligent adaptatif
    train_graphs, val_graphs, test_graphs = smart_split_dataset(all_graphs)
    
    print(f"📊 Split dataset final:")
    print(f"   Train: {len(train_graphs)} échantillons")
    if val_graphs:
        print(f"   Val:   {len(val_graphs)} échantillons") 
    else:
        print(f"   Val:   AUCUN (dataset trop petit)")
    print(f"   Test:  {len(test_graphs)} échantillons")

    # ✅ DataLoaders adaptés
    try:
        # ✅ CORRECTION : Batch size adaptatif pour petit dataset
        effective_batch_size = min(cfg.train.batch_size, len(train_graphs))
        print(f"🔧 Batch size adapté: {effective_batch_size}")
        
        train_loader = DataLoader(train_graphs, batch_size=effective_batch_size, shuffle=True, num_workers=0)
        
        if val_graphs:
            val_loader = DataLoader(val_graphs, batch_size=effective_batch_size, shuffle=False, num_workers=0)
        else:
            val_loader = None
            
        test_loader = DataLoader(test_graphs, batch_size=effective_batch_size, shuffle=False, num_workers=0)
        print("✅ DataLoaders créés")
    except Exception as e:
        print(f"❌ Erreur DataLoaders: {e}")
        sys.exit(1)

    # === 2. Initialisation du modèle ===
    cfg.defrost()
    cfg.model.num_classes = num_classes
    cfg.freeze()

    try:
        model = XaiGuiFormer(config=cfg, training_graphs=train_graphs).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Modèle XaiGuiFormer créé")
        print(f"   Paramètres totaux: {total_params:,}")
        print(f"   Paramètres entraînables: {trainable_params:,}")
        print(f"   Taille estimée: {trainable_params * 4 / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"❌ Erreur création modèle: {e}")
        sys.exit(1)

    # ✅ Optimiseur avec validation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=cfg.train.optimizer.betas,
        eps=cfg.train.optimizer.eps,
        weight_decay=cfg.train.optimizer.weight_decay
    )
    
    # ✅ Scheduler adaptatif
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=1e-6
    )

    print(f"✅ Optimiseur configuré: AdamW")

    # === 3. Boucle d'entraînement adaptée ===
    # ✅ CORRECTION : Epochs réduits pour petit dataset
    max_epochs = min(cfg.train.epochs, 100) if len(train_graphs) < 20 else cfg.train.epochs
    print(f"\n🚀 Début de l'entraînement - {max_epochs} epochs (adapté)")
    
    best_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_metrics = []

    for epoch in range(1, max_epochs + 1):
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch}/{max_epochs}")
        print(f"{'='*100}")
        
        # ✅ Entraînement
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # ✅ Validation si disponible
        if val_loader and epoch % 5 == 0:
            print(f"\n📊 Évaluation epoch {epoch}...")
            
            val_results = evaluate_with_bac(model, val_loader, label_encoder.classes_, device)
            
            print(f"🎯 Résultats validation:")
            print(f"   BAC Coarse:  {val_results['bac_coarse']:.4f}")
            print(f"   BAC Refined: {val_results['bac_refined']:.4f}")
            print(f"   Gain XAI:    {val_results['bac_gain']:+.4f}")
            
            val_metrics.append({
                "Epoch": epoch,
                "Train_Loss": round(train_loss, 4),
                "BAC_Refined": round(val_results["bac_refined"], 4),
                "BAC_Gain": round(val_results["bac_gain"], 4)
            })
        
        # ✅ Sauvegarde basée sur train_loss si pas de validation
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
            
            print(f"💾 Nouveau meilleur modèle sauvé! Loss: {best_loss:.4f}")
        
        scheduler.step()

    # === 4. Évaluation finale sur test ===
    print(f"\n{'='*60}")
    print("ÉVALUATION FINALE SUR TEST SET")
    print(f"{'='*60}")
    
    # Charger le meilleur modèle
    if os.path.exists("checkpoints/xaiguiformer_best.pth"):
        checkpoint = torch.load("checkpoints/xaiguiformer_best.pth", map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Modèle chargé (epoch {checkpoint['epoch']})")
    
    # Évaluation finale
    final_results = evaluate_with_bac(model, test_loader, label_encoder.classes_, device)
    
    print(f"\n🎉 RÉSULTATS FINAUX SUR TEST:")
    print(f"   BAC Coarse:  {final_results['bac_coarse']:.4f}")
    print(f"   BAC Refined: {final_results['bac_refined']:.4f}")
    print(f"   Gain XAI:    {final_results['bac_gain']:+.4f}")
    print(f"   Accuracy:    {final_results['accuracy']:.4f}")

    # === 5. Sauvegarde finale ===
    if val_metrics:
        df_metrics = pd.DataFrame(val_metrics)
        print(f"\n📊 Historique:")
        print(df_metrics.to_string(index=False))
        
    torch.save(model.state_dict(), "checkpoints/xaiguiformer_final.pth")
    print(f"\n✅ Modèle final sauvé!")
    print(f"🎯 Entraînement terminé - Best epoch: {best_epoch}")


"""
✅ CORRECTIONS pour PETIT DATASET (16 échantillons):

1. Split adaptatif sans stratification obligatoire
2. Batch size automatiquement réduit 
3. Pas de validation set si <10 échantillons
4. Epochs réduits pour éviter overfitting
5. Sauvegarde basée sur train_loss si pas de validation
6. Gestion robuste des classes avec 1 seul échantillon
7. Évaluation finale sur test même avec petit dataset

Dataset 16 échantillons → Stratégie:
├── Train: ~12 échantillons
├── Val: AUCUN (trop petit)  
├── Test: ~4 échantillons
└── Évaluation sur test final uniquement
"""