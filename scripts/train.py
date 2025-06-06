"""
Script d’entraînement pour un modèle GNN (EEGConnectomeGNN) à partir des
graphes de connectomes EEG pré-calculés et sauvegardés dans un fichier .pkl.

Les graphes sont chargés, séparés en train/test, puis utilisés dans un DataLoader
pour entraîner un classifieur supervisé sur plusieurs epochs.
"""

import torch
from torch_geometric.data import DataLoader
import pickle
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from model.trainer import EEGConnectomeGNN
from model.losses import compute_class_weights, weighted_cross_entropy_loss
from model.evaluator import evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_graph_dataset(pickle_path):
    """
    Charge un dataset de graphes depuis un fichier .pkl.

    Args:
        pickle_path (str): Chemin du fichier contenant les graphes (format pickle).

    Returns:
        list: Liste d'objets torch_geometric.data.Data (un graphe par sujet/session).
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def train(model, loader, optimizer, criterion):
    """
    Fonction d’entraînement pour une epoch.

    Args:
        model (torch.nn.Module): Modèle GNN à entraîner.
        loader (DataLoader): DataLoader contenant les graphes du jeu d'entraînement.
        optimizer (torch.optim.Optimizer): Optimiseur (ex. Adam).
        criterion (callable): Fonction de perte personnalisée.

    Returns:
        float: Perte moyenne sur l'epoch.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    # Charger les graphes depuis le fichier pickle
    all_graphs = load_graph_dataset("data/TDBRAIN/connectome/connectomes_graphs.pkl")

    # Vérification des labels présents
    all_labels = [data.y.item() for data in all_graphs]
    num_classes = len(set(all_labels))
    print(f"Nombre de classes détectées : {num_classes}") # remove

    all_metrics = []

    # Création du LabelEncoder pour récupérer les noms de classes
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)  # Pour permettre .classes_

    # Séparation train/test
    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16)

    # Calcul des poids de classes pour la perte pondérée
    class_weights = compute_class_weights(train_graphs)
    criterion = lambda outputs, targets: weighted_cross_entropy_loss(outputs, targets, class_weights.to(outputs.device))

    # Initialisation du modèle
    input_dim = train_graphs[0].x.shape[1]
    model = EEGConnectomeGNN(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train + éval
    for epoch in range(1, 21):
        loss = train(model, train_loader, optimizer, criterion)
        metrics = evaluate(model, test_loader, class_names=label_encoder.classes_, device="cpu", save_path="results", epoch=epoch)
        
        all_metrics.append({
            "Epoch": epoch,
            "Loss": loss,
            "Accuracy": metrics["accuracy"],
            "F1-macro": metrics["f1_macro"],
            "F1-weighted": metrics["f1_weighted"]
        })

    # Recap
    df_metrics = pd.DataFrame(all_metrics).round(4)
    print("\nRésumé des performances :\n")
    print(df_metrics.to_string(index=False))
