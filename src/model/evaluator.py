"""
evaluator.py

Évalue les performances du modèle XaiGuiFormer sur un ensemble de test.
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
    Évalue les performances du modèle XaiGuiFormer sur un DataLoader.

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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y_true = batch.y
            age = batch.age.view(-1, 1)
            gender = batch.gender.view(-1, 1)
            freq_bounds = batch.freq_bounds
            if freq_bounds.dim() == 1:
                freq_bounds = freq_bounds.unsqueeze(0)


            logits_coarse, logits_refined = model(batch, freq_bounds, age, gender, y_true=None)
            preds = logits_refined.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y_true.cpu().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_weighted = f1_score(all_targets, all_preds, average="weighted")
    labels_used = list(unique_labels(all_targets, all_preds))

    report = classification_report(
        all_targets,
        all_preds,
        labels=labels_used,
        target_names=[class_names[i] for i in labels_used],
        zero_division=0,
        output_dict=True
    )
    cm = confusion_matrix(all_targets, all_preds)

    result = {
        "epoch": int(epoch) if epoch is not None else None,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": _convert_report_to_serializable(report)
    }

    # Sauvegarde JSON si chemin fourni
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        filename = f"metrics_epoch_{epoch}.json" if epoch is not None else "metrics.json"
        with open(os.path.join(save_path, filename), "w") as f:
            json.dump(result, f, indent=4)

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
