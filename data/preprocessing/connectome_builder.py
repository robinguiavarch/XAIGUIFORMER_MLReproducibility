import os
import mne
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from config import get_cfg_defaults

from mne_connectivity import spectral_connectivity_epochs
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def compute_connectivity(epochs, method, fmin, fmax, sfreq):
    con = spectral_connectivity_epochs(
        epochs,
        method=method,
        mode='fourier',
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False,
    )
    return con.get_data(output='dense')[:, :, 0]


def build_connectomes(cfg):
    preprocessed_path = cfg.connectome.path.data_dir
    save_path = cfg.connectome.path.save_dir
    methods = cfg.connectome.methods
    freq_bands = cfg.connectome.frequency_band
    freq_bands = {k: list(v) for k, v in freq_bands.items()}
    
    print("Freq band test:", list(freq_bands.items())[0])
    print("Methods:", methods) # À retirer ensuite
    print("Frequency bands:", freq_bands) # À retirer ensuite

    for group in os.listdir(preprocessed_path):
        group_path = os.path.join(preprocessed_path, group)
        print(f"Processing group: {group}")
        if not os.path.isdir(group_path):
            print(f"Skipping {group_path}, not a directory.")
            continue

        for subj in tqdm(os.listdir(group_path), desc=f"Processing {group}"):
            subj_path = os.path.join(group_path, subj)
            print(f"Processing subject: {subj}")
            epoch_file = os.path.join(subj_path, f"{group}_{subj}_EC_epo.fif")

            if not os.path.exists(epoch_file):
                print(f"Epoch file {epoch_file} does not exist, skipping {subj}.")
                continue

            epochs = mne.read_epochs(epoch_file, preload=True) 
            print(f"{subj}: {len(epochs)} epochs loaded") # À retirer ensuite
            if len(epochs) == 0: # remove
                print(f"No epochs found for {subj}, skipping...") # remove
                continue # remove
            sfreq = epochs.info['sfreq']

            print("On arrive à la boucle...") # À retirer ensuite
            for method in methods:
                for band, f_range in freq_bands.items():
                    fmin, fmax = f_range
                    print(f"→ Computing {subj}, method={method}, band={band} ({fmin}-{fmax}Hz)") # À retirer ensuite

                    con_matrix = compute_connectivity(epochs, method, fmin, fmax, sfreq)
                    
                    if con_matrix is None: # remove
                        print(f"No connectivity matrix returned for {subj} ({method}, {band})") # remove
                        continue # remove
                    if con_matrix.shape[0] == 0: # remove
                        print(f"Empty matrix for {subj} ({method}, {band})") # remove
                        continue # remove

                    save_dir = os.path.join(save_path, group, subj)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    out_file = os.path.join(save_dir, f"{subj}_{method}_{band}.npy")
                    print(f"Saving: {out_file} (shape={con_matrix.shape})") # À retirer ensuite
                    np.save(out_file, con_matrix)

                    print(f"Connectome saved: {out_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build connectomes from preprocessed EEG data")
    parser.add_argument("--config", type=str, default="configs/tdbrain_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    os.makedirs(cfg.connectome.path.save_dir, exist_ok=True)
    build_connectomes(cfg)
