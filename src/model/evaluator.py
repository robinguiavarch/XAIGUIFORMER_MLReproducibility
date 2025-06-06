"""
evaluator.py

Contient les fonctions d'évaluation pour un modèle GNN entraîné sur
des graphes de connectomes EEG, en utilisant les métriques classiques :
accuracy, F1-score, matrice de confusion, etc.
Les résultats peuvent être sauvegardés dans un fichier JSON.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import json
import os


def evaluate(model, loader, class_names=None, device="cpu", save_path=None, epoch=None):
    """
    Évalue les performances du modèle sur un DataLoader et peut sauvegarder les métriques.

    Args:
        model (torch.nn.Module): Modèle GNN entraîné.
        loader (DataLoader): DataLoader contenant les graphes à évaluer.
        class_names (list, optional): Noms des classes pour l'affichage du rapport.
        device (str): Appareil ('cpu' ou 'cuda').
        save_path (str, optional): Dossier où sauvegarder les métriques en JSON.
        epoch (int, optional): Numéro de l'epoch (pour nommer les fichiers).

    Returns:
        dict: Dictionnaire contenant les métriques (accuracy, f1, etc.).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch.y.cpu().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_weighted = f1_score(all_targets, all_preds, average="weighted")
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)

    result = {
        "epoch": int(epoch) if epoch is not None else None,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": _convert_report_to_serializable(report)
    }

    return result


def _convert_report_to_serializable(report):
    """
    Convertit le rapport de classification en dictionnaire 100% JSON-serializable.

    Args:
        report (dict): Rapport de classification retourné par classification_report(..., output_dict=True).

    Returns:
        dict: Rapport avec clés/valeurs compatibles JSON.
    """
    converted = {}
    for key, value in report.items():
        new_key = str(key)
        if isinstance(value, dict):
            converted[new_key] = {str(k): float(v) for k, v in value.items()}
        else:
            converted[new_key] = float(value)
    return converted
