import torch
from torch.utils.data import DataLoader
from utils.data_transformer_tensor_timeseries import EEGFrequencyTokensDataset, FrequencyTokensData
from models.xaiguiformer_timeseries import XAIguiFormerTimeSeries, get_frequency_bands_tensor

def create_small_chunked_loader(root, chunk_size=2000, overlap=0.5, batch_size=2, n_patients=2):
    """
    Charge un mini dataset chunké pour test rapide.
    Retourne une DataLoader standard sur des dicts de tensors (as_dict=True)
    """
    base_dataset = EEGFrequencyTokensDataset(root, "TDBRAIN", "train")
    mini_dataset = [base_dataset[i] for i in range(min(n_patients, len(base_dataset)))]
    chunked_samples = []
    step_size = int(chunk_size * (1 - overlap))
    for sample in mini_dataset:
        freq = sample.frequency_tokens
        tlen = freq.shape[2]
        for start in range(0, tlen - chunk_size + 1, step_size):
            chunked = freq[:, :, start:start + chunk_size]
            # On stocke les chunks comme des dicts !
            chunked_samples.append({
                "frequency_tokens": chunked,
                "y": sample.y,
                "demographic_info": sample.demographic_info,
                "eid": f"{sample.eid}_chunk_{start}"
            })
    chunked_samples = chunked_samples[:4]  # Pour ne garder que quelques batches pour test rapide
    return DataLoader(chunked_samples, batch_size=batch_size, shuffle=True)

def test_forward_pass():
    freq_bands = get_frequency_bands_tensor()
    model = XAIguiFormerTimeSeries(
        num_channels=33,
        num_classes=4,
        freqband=freq_bands,
        num_kernels=200,
        attention_heads=2,
        output_features=128,
        num_heads=4,
        num_transformer_layers=12,
        mlp_ratio=4.0,
        init_values=0.001,
        dropout=0.1,
        attn_drop=0.0,
        droppath=0.0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_root = "data"  # adapte le chemin si besoin

    loader = create_small_chunked_loader(data_root, chunk_size=2000, overlap=0.5, batch_size=2, n_patients=2)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"--- Batch {i} ---")
            # batch est un dict de tensors
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            preds, preds_xai = model(batch)
            print("Vanilla:", preds.shape, "XAI:", preds_xai.shape)
            print("Vanilla preds (first row):", preds[0].cpu().numpy())
            print("OK forward.")
            break  # Un seul batch suffit pour le test

if __name__ == "__main__":
    test_forward_pass()
    print("✅ test_main_timeseries.py : test minimal pipeline OK")
