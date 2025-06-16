import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from torch_geometric.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Add src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config import get_cfg_defaults
from model.xaiguiformer_pipeline import XaiGuiFormer


# === Functions ===

def load_graph_dataset(pickle_path):
    """
    Load a pickled dataset of graph objects.

    Args:
        pickle_path (str): Path to the pickle file.

    Returns:
        list: A list of PyTorch Geometric graph data objects.
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def extract_attention_hook(attn_weights_list):
    """
    Create a forward hook to extract attention weights from a model layer.

    Args:
        attn_weights_list (list): A list to store extracted attention tensors.

    Returns:
        function: The forward hook function.
    """
    def hook(module, input, output):
        if hasattr(module, 'attn_weights_cache'):
            attn_weights_list.append(module.attn_weights_cache.detach().cpu())
    return hook


def plot_attention(attn_matrix, title, save_path):
    """
    Plot a heatmap of an attention matrix and save it to a file.

    Args:
        attn_matrix (Tensor): 2D tensor representing the attention scores.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_matrix, cmap='viridis', square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# === Main script ===

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(cfg.out_root, "attention_maps"), exist_ok=True)

    # === Load graph dataset
    graph_path = os.path.join(cfg.connectome.path.save_dir, "connectomes_graphs.pkl")
    all_graphs = load_graph_dataset(graph_path)
    test_graphs = all_graphs[-32:]  # small subset for debug/visualization
    loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # === Label encoding
    all_labels = [g.y.item() for g in all_graphs]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    cfg.model.num_classes = len(label_encoder.classes_)
    cfg.model.num_node_feat = test_graphs[0].x.shape[1]

    # === Load trained model
    model = XaiGuiFormer(config=cfg, training_graphs=all_graphs)
    checkpoint_path = os.path.join("checkpoints", "tdbrain_best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # === Register attention hooks
    vanilla_attns, xai_attns = [], []
    for layer in model.shared_transformer.layers:
        # Capture standard and XAI-guided attention (if present) using hooks
        layer.register_forward_hook(extract_attention_hook(vanilla_attns))

    # === Forward pass + attention visualization
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        age = batch.age.view(-1, 1)
        gender = batch.gender.view(-1, 1)
        freq_bounds = batch.freq_bounds[0]

        with torch.no_grad():
            model(batch, freq_bounds, age, gender, y_true=batch.y)

        if vanilla_attns:
            attn_tensor = vanilla_attns[-1][0].mean(dim=0)  
            plot_attention(
                attn_tensor,
                f"Final Attention Layer - Sample {i}",
                os.path.join(cfg.out_root, "attention_maps", f"attn_sample_{i}.png")
            )

        # Clear for next sample
        vanilla_attns.clear()
