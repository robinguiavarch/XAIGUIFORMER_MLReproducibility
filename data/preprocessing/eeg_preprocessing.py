import os
import mne
import numpy as np
import mne_icalabel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from config import get_cfg_defaults
from pyprep.prep_pipeline import PrepPipeline

def custom_read_raw_brainvision(fname, montage, phenotype, preload=True):
    EEG_raw = mne.io.read_raw_brainvision(fname, preload=preload)
    channel_type = {'VPVA': 'eog', 'VNVB': 'eog',
                    'HPHL': 'eog', 'HNHR': 'eog',
                    'Erbs': 'ecg', 'OrbOcc': 'eog',
                    'Mass': 'emg'}
    EEG_raw.set_channel_types(channel_type)
    EEG_raw.set_montage(montage)
    EEG_raw.custom_events = mne.make_fixed_length_events(EEG_raw, id=30, start=0, stop=120, duration=40, overlap=20)
    EEG_raw.custom_event_id = {'eyes close': 30}
    EEG_raw.info['description'] = phenotype
    return EEG_raw

def preprocessing(EEG_raw, config, verbose=True):
    n_jobs = config.preprocessing.n_jobs
    n_step = 0

    n_step += 1
    if verbose:
        print(f'Step {n_step}: Detecting bad channels...')
    prep_params = {"ref_chs": "eeg", "reref_chs": "eeg",
                   "line_freqs": np.arange(config.preprocessing.line_frequency, EEG_raw.info['sfreq'] / 2, config.preprocessing.line_frequency)}
    filter_kwargs = {"method": "fir", "n_jobs": n_jobs}
    noisy_detector = PrepPipeline(EEG_raw, prep_params, EEG_raw.get_montage(),
                                  channel_wise=True, filter_kwargs=filter_kwargs)
    noisy_detector.fit()
    EEG_raw.info['bads'] = noisy_detector.interpolated_channels

    n_step += 1
    if verbose:
        print(f'Step {n_step}: Filtering...')
    EEG_bandpass = EEG_raw.filter(l_freq=config.preprocessing.l_freq, h_freq=config.preprocessing.h_freq, n_jobs=n_jobs)

    n_step += 1
    if verbose:
        print(f'Step {n_step}: Segmenting into epochs...')
    picked_events_onset = EEG_bandpass.custom_events
    if abs(EEG_bandpass.info['sfreq'] - 500) < 1e-6 or abs(EEG_bandpass.info['sfreq'] - 512) < 1e-6:
        EEG_eyes_close = mne.Epochs(EEG_bandpass, picked_events_onset, {'eyes close': 30}, baseline=None, tmin=5, tmax=35,
                                    preload=True, reject=None, proj=False, decim=2)
    else:
        EEG_eyes_close = mne.Epochs(EEG_bandpass, picked_events_onset, {'eyes close': 30}, baseline=None, tmin=5, tmax=35,
                                    preload=True, reject=None, proj=False)

    n_step += 1
    if verbose:
        print(f'Step {n_step}: Applying ICA...')
    nb_channels = len(mne.pick_types(EEG_eyes_close.info, eeg=True, exclude='bads'))
    ica = mne.preprocessing.ICA(n_components=min(20, nb_channels),
                                method=config.preprocessing.ica.method,
                                fit_params=dict(config.preprocessing.ica.fit_params))
    ica.fit(EEG_eyes_close)
    ica.exclude = []
    ic_info = mne_icalabel.label_components(EEG_eyes_close, ica, method='iclabel')
    ic_probability, ic_labels = ic_info['y_pred_proba'], ic_info["labels"]
    ica.exclude = [idx for idx, label in enumerate(ic_labels) if label not in ["brain", "other"] and ic_probability[idx] >= config.preprocessing.ica.ic_label_threshold]
    print(f"Excluding ICA components: {np.array(ic_labels)[ica.exclude]}")
    EEG_ica = ica.apply(EEG_eyes_close)

    n_step += 1
    if verbose:
        print(f'Step {n_step}: Interpolating bad channels...')
    EEG_interp = EEG_ica.interpolate_bads()

    n_step += 1
    if verbose:
        print(f'Step {n_step}: Setting average reference...')
    return EEG_interp.set_eeg_reference(ref_channels='average', ch_type='eeg')

def preprocessingPipeline(fname, montage, phenotype, config, save_path):
    EEG_raw = custom_read_raw_brainvision(fname, montage, phenotype)
    if EEG_raw.info['description'] != 'bad':
        EEG_prep = preprocessing(EEG_raw, config, verbose=False)
        if not os.path.exists(save_path):
            dir, _ = os.path.split(save_path)
            os.makedirs(dir, exist_ok=True)
        EEG_prep.save(save_path, fmt='double', overwrite=True)
    else:
        print(f'({phenotype}) Data labeled as bad — skipping.')

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/tdbrain_config.yaml')
    cfg.freeze()

    f_eleloc = cfg.preprocessing.path.electrode_locations
    data_dir = cfg.preprocessing.path.data_dir
    save_dir = cfg.preprocessing.path.save_dir

    for subj_dir in os.listdir(data_dir):
        subj_path = os.path.join(data_dir, subj_dir)
        if not subj_dir.startswith("sub-") or not os.path.isdir(subj_path):
            continue

        ses_path = os.path.join(subj_path, "ses-1", "eeg")
        if not os.path.exists(ses_path):
            continue

        for file in os.listdir(ses_path):
            if file.endswith("_task-restEC_eeg.vhdr"):
                file_base = file.replace("_task-restEC_eeg.vhdr", "")
                fname = os.path.join(ses_path, file)
                montage = mne.channels.make_standard_montage("standard_1020")
                save_path = os.path.join(save_dir, subj_dir, "ses-1", f'{file_base}_EC_epo.fif')
                preprocessingPipeline(fname, montage, file_base, cfg, save_path)


