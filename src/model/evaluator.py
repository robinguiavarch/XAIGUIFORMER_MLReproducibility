"""
evaluator.py

✅ CORRIGÉ pour l'architecture finale XaiGuiFormer avec shared transformer.
Évalue les performances sur un ensemble de test avec gestion correcte du batching PyG.
Supporte les prédictions coarse et refined, et calcule les métriques principales :
accuracy, F1-score (macro/weighted), matrice de confusion et classification report.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import json
import os
import warnings

# Ignore les avertissements de Captum liés aux hooks non linéaires
warnings.filterwarnings("ignore", message="Setting forward, backward hooks and attributes on non-linear")


def evaluate(model, loader, class_names=None, device="cpu", save_path=None, epoch=None):
    """
    ✅ CORRIGÉ : Évalue les performances du modèle XaiGuiFormer sur un DataLoader.

    Args:
        model (torch.nn.Module): Modèle complet (XaiGuiFormer).
        loader (DataLoader): DataLoader contenant les batchs PyG.
        class_names (list, optional): Liste des noms des classes.
        device (str): Appareil d'exécution ('cpu' ou 'cuda').
        save_path (str, optional): Dossier où sauvegarder les résultats.
        epoch (int, optional): Numéro d'époque (pour la sauvegarde).

    Returns:
        dict: Dictionnaire de métriques.
    """
    model.eval()
    all_preds_coarse = []
    all_preds_refined = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # ✅ CORRECTION : Gestion correcte du batching PyG
            B = batch.y.shape[0]
            
            # Reshape si nécessaire (PyG concatène les tenseurs)
            if hasattr(batch, 'x_tokens') and batch.x_tokens.dim() == 2:
                total_tokens = batch.x_tokens.shape[0]
                d = batch.x_tokens.shape[1]
                Freq = total_tokens // B
                
                # Reshape x_tokens et freq_bounds
                batch.x_tokens = batch.x_tokens.view(B, Freq, d)
                
                # freq_bounds peut être [total_tokens, 2] → [B, Freq, 2]
                if batch.freq_bounds.shape[0] == total_tokens:
                    batch.freq_bounds = batch.freq_bounds.view(B, Freq, 2)
            
            # ✅ Extraire les données correctement
            y_true = batch.y
            age = batch.age.view(-1, 1)        # [B, 1]
            gender = batch.gender.view(-1, 1)  # [B, 1]
            
            # freq_bounds : prendre le premier échantillon (identiques)
            if batch.freq_bounds.dim() == 3:
                freq_bounds = batch.freq_bounds[0]  # [Freq, 2]
            else:
                freq_bounds = batch.freq_bounds
                
            if freq_bounds.dim() == 1:
                freq_bounds = freq_bounds.view(-1, 2)

            # ✅ FORWARD PASS : retourne (logits_coarse, logits_refined)
            try:
                logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
                
                # Prédictions
                preds_coarse = logits_coarse.argmax(dim=1)
                preds_refined = logits_refined.argmax(dim=1)
                
                all_preds_coarse.extend(preds_coarse.cpu().tolist())
                all_preds_refined.extend(preds_refined.cpu().tolist())
                all_targets.extend(y_true.cpu().tolist())
                
            except Exception as e:
                print(f"⚠️  Erreur lors de l'évaluation du batch: {e}")
                continue

    # ✅ Calcul des métriques pour REFINED (performance principale)
    if not all_preds_refined:
        print("❌ Aucune prédiction générée")
        return {}
        
    accuracy_refined = accuracy_score(all_targets, all_preds_refined)
    f1_macro_refined = f1_score(all_targets, all_preds_refined, average="macro", zero_division=0)
    f1_weighted_refined = f1_score(all_targets, all_preds_refined, average="weighted", zero_division=0)
    
    # ✅ Calcul des métriques pour COARSE (comparaison)
    accuracy_coarse = accuracy_score(all_targets, all_preds_coarse)
    f1_macro_coarse = f1_score(all_targets, all_preds_coarse, average="macro", zero_division=0)
    
    # Labels utilisés
    labels_used = list(unique_labels(all_targets, all_preds_refined))
    
    # Classification report pour refined
    if class_names:
        target_names = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in labels_used]
    else:
        target_names = [f"Class_{i}" for i in labels_used]
        
    report = classification_report(
        all_targets,
        all_preds_refined,
        labels=labels_used,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    
    # Matrice de confusion pour refined
    cm = confusion_matrix(all_targets, all_preds_refined)

    # ✅ Résultats complets avec comparaison coarse vs refined
    result = {
        "epoch": int(epoch) if epoch is not None else None,
        "refined": {
            "accuracy": float(accuracy_refined),
            "f1_macro": float(f1_macro_refined),
            "f1_weighted": float(f1_weighted_refined),
            "confusion_matrix": cm.tolist(),
            "classification_report": _convert_report_to_serializable(report)
        },
        "coarse": {
            "accuracy": float(accuracy_coarse),
            "f1_macro": float(f1_macro_coarse)
        },
        "improvement": {
            "accuracy_gain": float(accuracy_refined - accuracy_coarse),
            "f1_macro_gain": float(f1_macro_refined - f1_macro_coarse)
        },
        "samples_evaluated": len(all_targets)
    }

    # ✅ Affichage des résultats
    print(f"📊 Évaluation (epoch {epoch}):")
    print(f"   Échantillons: {len(all_targets)}")
    print(f"   COARSE   - Acc: {accuracy_coarse:.4f}, F1: {f1_macro_coarse:.4f}")
    print(f"   REFINED  - Acc: {accuracy_refined:.4f}, F1: {f1_macro_refined:.4f}")
    print(f"   GAIN     - Acc: {accuracy_refined - accuracy_coarse:+.4f}, F1: {f1_macro_refined - f1_macro_coarse:+.4f}")

    # Sauvegarde JSON si chemin fourni
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"metrics_epoch_{epoch}.json" if epoch is not None else "metrics.json"
        with open(os.path.join(save_path, filename), "w") as f:
            json.dump(result, f, indent=4)
        print(f"   ✅ Sauvé: {os.path.join(save_path, filename)}")

    return result


def _convert_report_to_serializable(report):
    """
    Convertit le rapport de classification sklearn en dictionnaire 100% JSON-serializable.

    Args:
        report (dict): Classification report de sklearn (output_dict=True).

    Returns:
        dict: Rapport sérialisable.
    """
    converted = {}
    for key, value in report.items():
        new_key = str(key)
        if isinstance(value, dict):
            converted[new_key] = {str(k): float(v) for k, v in value.items()}
        else:
            converted[new_key] = float(value)
    return converted


def compute_balanced_accuracy(y_true, y_pred):
    """
    ✅ NOUVEAU : Calcule la Balanced Accuracy (BAC) comme dans l'article
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        
    Returns:
        float: Balanced Accuracy
    """
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


def evaluate_with_bac(model, loader, class_names=None, device="cpu"):
    """Évaluation rapide avec BAC"""
    model.eval()
    all_preds_coarse = []
    all_preds_refined = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # ✅ NOUVEAU: Gestion batch basée sur votre diagnostic
            B = batch.age.shape[0]  # Age donne la vraie batch size
            
            y_true = batch.y
            age = batch.age.view(-1, 1)
            gender = batch.gender.view(-1, 1)
            
            # freq_bounds: déduire depuis x_tokens
            total_tokens = batch.x_tokens.shape[0]
            Freq = total_tokens // B
            
            # Prendre freq_bounds du premier échantillon (assumé identique)
            freq_bounds = batch.freq_bounds[:Freq]  # [18, 2]
            
            try:
                logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
                
                all_preds_coarse.extend(logits_coarse.argmax(dim=1).cpu().tolist())
                all_preds_refined.extend(logits_refined.argmax(dim=1).cpu().tolist())
                all_targets.extend(y_true.cpu().tolist())
                
            except Exception as e:
                print(f"⚠️  Erreur batch: {e}")
                continue

    if not all_preds_refined:
        return {
            "bac_coarse": 0.0, 
            "bac_refined": 0.0, 
            "accuracy": 0.0,
            "bac_gain": 0.0  # ✅ AJOUT CRITIQUE
        }
        
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    
    bac_coarse = balanced_accuracy_score(all_targets, all_preds_coarse)
    bac_refined = balanced_accuracy_score(all_targets, all_preds_refined)
    accuracy = accuracy_score(all_targets, all_preds_refined)
    
    return {
        "bac_coarse": float(bac_coarse),
        "bac_refined": float(bac_refined), 
        "accuracy": float(accuracy),
        "bac_gain": float(bac_refined - bac_coarse)  # ✅ CALCUL CORRECT
    }


"""
✅ CORRECTIONS MAJEURES dans evaluator.py:

1. Gestion correcte du batching PyG (reshape x_tokens et freq_bounds)
2. Forward pass retourne (logits_coarse, logits_refined) 
3. Évaluation des DEUX sorties (coarse vs refined)
4. Calcul du gain XAI (improvement)
5. Métriques BAC conformes à l'article
6. Gestion robuste des erreurs
7. Affichage informatif des résultats
8. Sauvegarde JSON structurée

Architecture Flow:
├── Input: PyG batch avec x_tokens, freq_bounds, age, gender, y
├── Reshape: Gestion du batching PyG (concaténation → tensor 3D)
├── Forward: model(batch, freq_bounds, age, gender) → (logits_coarse, logits_refined)
├── Metrics: Évaluation coarse vs refined + gain XAI
└── Output: Résultats détaillés avec comparaison performances
"""