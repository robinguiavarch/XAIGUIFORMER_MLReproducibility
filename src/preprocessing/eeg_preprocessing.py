"""
EEG preprocessing script for CSV-formatted RestEC (eyes closed) data.
Implements PREP pipeline, filtering, ICA, epoching, and average referencing.
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
import mne_icalabel
from pyprep.prep_pipeline import PrepPipeline

# === Configuration import ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from config import get_cfg_defaults


def custom_read_raw_csv(fname, montage, phenotype, preload=True):
    """
    Charge un fichier EEG au format CSV avec les bons types de canaux,
    le montage EEG standard et crée des événements "yeux fermés" artificiels.
    """
    import pandas as pd
    from mne import create_info
    from mne.io import RawArray

    df = pd.read_csv(fname)
    ch_names = df.columns.tolist()
    sfreq = 250  # <- À adapter si ce n’est pas correct
    data = df.values.T

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = RawArray(data, info)

    # Marquer les canaux spéciaux comme non-EEG
    special_types = {
        'VPVA': 'eog', 'VNVB': 'eog',
        'HPHL': 'eog', 'HNHR': 'eog',
        'Erbs': 'ecg', 'OrbOcc': 'eog',
        'Mass': 'emg'
    }

    # Vérifie lesquels de ces canaux sont présents et change leur type
    present_specials = {k: v for k, v in special_types.items() if k in raw.ch_names}
    raw.set_channel_types(present_specials)

    # Montage seulement pour les canaux EEG
    raw.set_montage(montage, on_missing='ignore')  # on_missing=‘ignore’ pour éviter l’erreur
    raw.custom_events = mne.make_fixed_length_events(raw, id=30, start=0, stop=120, duration=40, overlap=20)
    raw.custom_event_id = {'eyes close': 30}
    raw.info['description'] = phenotype
    return raw



def preprocessing(raw, config, verbose=True):
    """
    Full preprocessing pipeline (PREP, filtering, epoching, ICA, reref).

    Args:
        raw (mne.io.Raw): Raw EEG object.
        config: yacs config.
        verbose (bool): Show progress.

    Returns:
        raw: Preprocessed MNE Epochs object.
    """
    n_jobs = config.preprocessing.n_jobs
    n_step = 0

    # === Step 1: PREP pipeline (bad channel interpolation)
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Detecting bad channels...")
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(
            config.preprocessing.line_frequency,
            raw.info['sfreq'] / 2,
            config.preprocessing.line_frequency
        )
    }
    prep = PrepPipeline(raw, prep_params, raw.get_montage(), channel_wise=True,
                        filter_kwargs={"method": "fir", "n_jobs": n_jobs})
    prep.fit()
    raw.info['bads'] = prep.interpolated_channels

    # === Step 2: Bandpass filter
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Bandpass filtering...")
    raw.filter(l_freq=config.preprocessing.l_freq,
               h_freq=config.preprocessing.h_freq, n_jobs=n_jobs)

    # === Step 3: Resample
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Resampling to 250 Hz...")
    raw = raw.resample(250)

    # === Step 4: Epoching eyes closed
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Creating epochs (eyes closed)...")
    events = raw.custom_events
    event_id = raw.custom_event_id
    tmin, tmax = 5, 35
    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, proj=False)

    # === Step 5: ICA + ICLabel
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Running ICA + ICLabel...")
    n_channels = len(mne.pick_types(epochs.info, eeg=True, exclude='bads'))
    ica = mne.preprocessing.ICA(
        n_components=min(20, n_channels),
        method=config.preprocessing.ica.method,
        fit_params=dict(config.preprocessing.ica.fit_params)
    )
    ica.fit(epochs)
    ic_info = mne_icalabel.label_components(epochs, ica, method='iclabel')
    probs, labels = ic_info['y_pred_proba'], ic_info["labels"]
    ica.exclude = [
        idx for idx, label in enumerate(labels)
        if label not in ["brain", "other"]
        and probs[idx] >= config.preprocessing.ica.ic_label_threshold
    ]
    print(f"Excluding ICA components: {np.array(labels)[ica.exclude]}")
    raw_ica = ica.apply(epochs)

    # === Step 6: Interpolate
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Interpolating...")
    raw_ica = raw_ica.interpolate_bads()

    # === Step 7: Average reference
    n_step += 1
    if verbose:
        print(f"[Step {n_step}] Re-referencing...")
    return raw_ica.set_eeg_reference(ref_channels='average', ch_type='eeg')


def preprocessingPipeline(fname, montage, phenotype, config, save_path):
    """
    Run the full preprocessing pipeline from a CSV RestEC file.

    Args:
        fname: Path to CSV.
        montage: MNE DigMontage.
        phenotype: Subject ID string.
        config: Global config.
        save_path: Where to write .fif
    """
    raw = custom_read_raw_csv(fname, montage, phenotype)
    if raw.info['description'] != 'bad':
        processed = preprocessing(raw, config, verbose=False)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        processed.save(save_path, fmt='double', overwrite=True)
    else:
        print(f"[{phenotype}] Data labeled as bad — skipped.")


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/tdbrain_config.yaml')
    cfg.freeze()

    data_dir = os.path.join(cfg.root, "raw", "derivatives")
    save_dir = cfg.preprocessing.path.save_dir
    montage = mne.channels.make_standard_montage("standard_1020")

    for subj_dir in os.listdir(data_dir):
        subj_path = os.path.join(data_dir, subj_dir)
        if not subj_dir.startswith("sub-") or not os.path.isdir(subj_path):
            continue

        ses_path = os.path.join(subj_path, "ses-1", "eeg")
        if not os.path.exists(ses_path):
            continue

        for file in os.listdir(ses_path):
            if file.endswith("_task-restEC_eeg.csv"):
                file_base = file.replace("_task-restEC_eeg.csv", "")
                fname = os.path.join(ses_path, file)
                save_path = os.path.join(save_dir, subj_dir, "ses-1", f'{file_base}_EC_epo.fif')
                preprocessingPipeline(fname, montage, file_base, cfg, save_path)
