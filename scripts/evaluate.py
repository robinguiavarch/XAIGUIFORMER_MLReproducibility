import torch
from torch_geometric.data import DataLoader
import os
import pickle
import sys
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer
from model.evaluator import evaluate

def load_graph_dataset(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def load_model(checkpoint_path, config, training_graphs):
    model = XaiGuiFormer(config=config, training_graphs=training_graphs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model


if __name__ == "__main__":
    # === Configuration ===
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. Chargement des graphes test ===
    test_path = os.path.join(cfg.connectome.path.save_dir, "connectomes_graphs_test.pkl")
    test_graphs = load_graph_dataset(test_path)
    test_loader = DataLoader(test_graphs, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers)

    # === 2. Encodage des classes ===
    all_labels = [g.y.item() for g in test_graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(label_encoder.classes_)

    # === 3. Compléter les valeurs dynamiques dans cfg.model ===
    cfg.model.num_classes = num_classes
    cfg.model.num_node_feat = test_graphs[0].x.shape[1]

    # === 4. Chargement du modèle entraîné ===
    model_path = os.path.join("checkpoints", "tdbrain_best_model.pth")
    model = load_model(model_path, cfg, training_graphs=test_graphs)
    model.to(device)
    model.eval()

    # === 5. Évaluation ===
    metrics = evaluate(
        model=model,
        loader=test_loader,
        class_names=label_encoder.classes_,
        device=device,
        save_path=os.path.join(cfg.out_root, "results"),
        epoch=None
    )

    # === 6. Affichage + sauvegarde ===
    print("\n\u2705 Performances sur le jeu de test:\n")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    os.makedirs(os.path.join(cfg.out_root, "results"), exist_ok=True)
    with open(os.path.join(cfg.out_root, "results", "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    df_summary = pd.DataFrame({
        "Accuracy": [metrics["accuracy"]],
        "F1-macro": [metrics["f1_macro"]],
        "F1-weighted": [metrics["f1_weighted"]]
    })
    df_summary.to_csv(os.path.join(cfg.out_root, "results", "eval_summary.csv"), index=False)
