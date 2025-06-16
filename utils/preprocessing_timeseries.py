print("=== SCRIPT STARTED ===")
"""
Preprocessing pipeline for XaiGuiFormer_TimeSeries
Step 1: TDBRAIN dataset reduction and organization using pre-selected patients
Step 2: Eyes Open cleaning
Step 3: Intelligent downsampling with new output directory
Step 4: Frequency band tokenization (NEW)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import shutil
from typing import Dict, List, Tuple, Optional
from scipy.signal import butter, filtfilt


class TDBRAINDatasetReducer:
    """
    Class to reduce and organize TDBRAIN dataset using pre-created 90 patients dataset
    """
    
    def __init__(self, 
                 tdbrain_root: str,
                 participants_csv: str,
                 target_root: str,
                 preselected_dataset: str = None):
        """
        Initialize TDBRAIN dataset reducer
        
        Args:
            tdbrain_root: Path to TDBRAIN-dataset-derivatives/
            participants_csv: Path to TDBRAIN_participants_V2.csv (original)
            target_root: Output path for reduced dataset
            preselected_dataset: Path to TDBRAIN_90_patients.csv (pre-selected)
        """
        self.tdbrain_root = Path(tdbrain_root)
        self.participants_csv = Path(participants_csv)
        self.target_root = Path(target_root)
        self.preselected_dataset = Path(preselected_dataset) if preselected_dataset else None
        
        self.derivatives_path = self.tdbrain_root / "derivatives"
        
        # Condition mapping for TDBRAIN (4 classes maintenant)
        self.condition_mapping = {
            'ADHD': 0,
            'MDD': 1, 
            'SMC': 2,
            'HEALTHY': 3
        }
        
        # Gender mapping: TDBRAIN 1=Male, 0=Female -> normalize to 0=Female, 1=Male
        self.gender_mapping = {1: 1, 0: 0}  # Keep as is since CSV already has 0=Female, 1=Male
    
    def load_preselected_dataset(self) -> pd.DataFrame:
        """Load the pre-selected 90 patients dataset"""
        print(f"Loading pre-selected dataset from: {self.preselected_dataset}")
        if not self.preselected_dataset.exists():
            print(f"ERROR: Pre-selected dataset not found at {self.preselected_dataset}")
            return pd.DataFrame()
        
        selected_df = pd.read_csv(self.preselected_dataset)
        print(f"Loaded {len(selected_df)} pre-selected patients")
        
        # Vérifier les colonnes nécessaires
        required_cols = ['participants_ID', 'indication', 'age', 'gender', 'split']
        missing_cols = [col for col in required_cols if col not in selected_df.columns]
        if missing_cols:
            print(f"ERROR: Missing columns in pre-selected dataset: {missing_cols}")
            return pd.DataFrame()
        
        return selected_df
    
    def scan_eeg_files(self) -> List[Dict]:
        """
        Scan EEG files to identify available patients (session 1 only)
        
        Structure: derivatives/sub-XXXXXXXX/ses-1/eeg/sub-XXXXXXXX_ses-1_task-restEC_eeg.csv
        """
        print(f"Scanning EEG files in: {self.derivatives_path}")
        if not self.derivatives_path.exists():
            print(f"ERROR: Derivatives path not found at {self.derivatives_path}")
            return []
        
        eeg_files = []
        patient_dirs = list(self.derivatives_path.iterdir())
        print(f"Found {len(patient_dirs)} directories in derivatives")
        
        valid_patient_count = 0
        for patient_dir in patient_dirs:
            if not patient_dir.is_dir() or not patient_dir.name.startswith('sub-'):
                continue
            valid_patient_count += 1
                
            patient_id = patient_dir.name  # e.g., 'sub-19690494'
            participant_id = patient_id.replace('sub-', '')  # e.g., '19690494'
            
            # Only check session 1
            session_dir = patient_dir / "ses-1"
            if not session_dir.exists():
                continue
                
            eeg_dir = session_dir / "eeg"
            if not eeg_dir.exists():
                continue
            
            # Look for Eyes Closed file (session 1 only)
            ec_file = eeg_dir / f"{patient_id}_ses-1_task-restEC_eeg.csv"
            eo_file = eeg_dir / f"{patient_id}_ses-1_task-restEO_eeg.csv"
            
            if ec_file.exists():
                eeg_files.append({
                    'patient_id': patient_id,
                    'participant_id': participant_id,
                    'session_id': 1,
                    'ec_file_path': str(ec_file),
                    'eo_file_path': str(eo_file) if eo_file.exists() else None,
                    'eeg_dir': str(eeg_dir)
                })
        
        print(f"Found {valid_patient_count} patient directories starting with 'sub-'")
        print(f"Found {len(eeg_files)} patients with valid EC files")
        if len(eeg_files) > 0:
            print(f"First few patients: {[f['patient_id'] for f in eeg_files[:3]]}")
        
        return eeg_files
    
    def merge_demographics_and_eeg(self, selected_df: pd.DataFrame, eeg_files: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Map pre-selected patients to EEG files and organize by splits
        
        Args:
            selected_df: Pre-selected 88 patients with splits
            eeg_files: Available EEG files
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        print(f"Merging pre-selected patients with EEG files...")
        print(f"Pre-selected patients: {len(selected_df)}")
        print(f"EEG files found: {len(eeg_files)}")
        
        if len(eeg_files) == 0:
            print("ERROR: No EEG files found!")
            return {}
        
        if len(selected_df) == 0:
            print("ERROR: No pre-selected patients loaded!")
            return {}
        
        # Convert EEG files to DataFrame
        eeg_df = pd.DataFrame(eeg_files)
        
        # Match EEG files with pre-selected patients
        matched_patients = []
        
        for _, patient_row in selected_df.iterrows():
            participant_id = str(patient_row['participants_ID'])
            
            # Vérifier si participant_id contient déjà 'sub-' ou pas
            if participant_id.startswith('sub-'):
                # Déjà au bon format
                full_patient_id = participant_id
                clean_participant_id = participant_id.replace('sub-', '')
            else:
                # Ajouter le préfixe 'sub-'
                full_patient_id = f"sub-{participant_id}"
                clean_participant_id = participant_id
            
            # Find matching EEG file
            matching_eeg = eeg_df[eeg_df['patient_id'] == full_patient_id]
            
            if len(matching_eeg) > 0:
                eeg_info = matching_eeg.iloc[0]
                
                # Créer l'entrée patient complète
                patient_entry = {
                    'patient_id': eeg_info['patient_id'],
                    'participant_id': clean_participant_id,  # Sans préfixe sub-
                    'session_id': 1,
                    'age': float(patient_row['age']),
                    'gender': int(patient_row['gender']),  # Déjà normalisé (0=Female, 1=Male)
                    'condition': patient_row['indication'],
                    'condition_code': self.condition_mapping[patient_row['indication']],
                    'split': patient_row['split'],
                    'ec_file_path': eeg_info['ec_file_path'],
                    'eo_file_path': eeg_info.get('eo_file_path'),
                    'eeg_dir': eeg_info['eeg_dir']
                }
                matched_patients.append(patient_entry)
            else:
                print(f"WARNING: No EEG file found for participant {participant_id} (looking for {full_patient_id})")
        
        # Convertir en DataFrame
        matched_df = pd.DataFrame(matched_patients)
        print(f"Successfully matched: {len(matched_df)} patients")
        
        if len(matched_df) == 0:
            print("ERROR: No patients matched between CSV and EEG files!")
            print("Check participant ID format")
            # Debug info
            print(f"Sample CSV IDs: {selected_df['participants_ID'].head().tolist()}")
            print(f"Sample EEG IDs: {[f['patient_id'] for f in eeg_files[:3]]}")
            return {}
        
        # Organiser par splits
        splits_data = {}
        for split in ['train', 'val', 'test']:
            split_data = matched_df[matched_df['split'] == split].copy()
            splits_data[split] = split_data
            print(f"  - {split}: {len(split_data)} patients")
        
        return splits_data
    
    def copy_eeg_files_to_reduced_dataset(self, splits_data: Dict[str, pd.DataFrame]) -> int:
        """
        Copy EEG Eyes Closed files to the reduced dataset structure
        
        Args:
            splits_data: Dictionary with train/val/test DataFrames
            
        Returns:
            Total number of files copied
        """
        print(f"📂 Copying EEG files to reduced dataset structure...")
        
        total_copied = 0
        
        for split_name, df in splits_data.items():
            split_dir = self.target_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            copied_count = 0
            
            for _, row in df.iterrows():
                ec_file_path = row['ec_file_path']
                
                if ec_file_path and Path(ec_file_path).exists():
                    # Générer le nom de fichier de destination
                    source_file = Path(ec_file_path)
                    dest_filename = f"{row['patient_id']}_ses-1_task-restEC_eeg.csv"
                    dest_path = split_dir / dest_filename
                    
                    try:
                        # Copier le fichier
                        shutil.copy2(source_file, dest_path)
                        copied_count += 1
                        total_copied += 1
                        
                        if copied_count % 10 == 0:
                            print(f"    Copied {copied_count} files to {split_name}...")
                        
                    except Exception as e:
                        print(f"  ❌ Erreur lors de la copie de {source_file.name}: {e}")
                else:
                    print(f"  ⚠️  Fichier EC introuvable pour {row['patient_id']}: {ec_file_path}")
            
            print(f"  ✅ {split_name}: {copied_count} fichiers copiés")
        
        print(f"📊 Total de fichiers EEG copiés: {total_copied}")
        return total_copied
    
    def create_reduced_dataset_structure(self, splits_data: Dict[str, pd.DataFrame]):
        """Create reduced dataset structure with metadata files and copied EEG files"""
        
        # Étape 1: Copier les fichiers EEG
        total_copied = self.copy_eeg_files_to_reduced_dataset(splits_data)
        
        # Étape 2: Créer les métadonnées avec les NOUVEAUX chemins
        for split_name in ['train', 'val', 'test']:
            split_dir = self.target_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata for each split avec nouveaux chemins
        for split_name, df in splits_data.items():
            split_dir = self.target_root / split_name
            metadata_file = split_dir / 'patients_metadata.json'
            
            patients_metadata = {}
            for _, row in df.iterrows():
                # Nouveau chemin dans le dataset réduit
                new_ec_path = split_dir / f"{row['patient_id']}_ses-1_task-restEC_eeg.csv"
                
                patients_metadata[row['patient_id']] = {
                    'participant_id': row['participant_id'],
                    'session_id': int(row['session_id']),
                    'age': float(row['age']),
                    'gender': int(row['gender']),  # 0=Female, 1=Male
                    'condition': row['condition'],
                    'condition_code': int(row['condition_code']),  # 0=ADHD, 1=MDD, 2=SMC, 3=HEALTHY
                    'ec_file_path': str(new_ec_path),  # NOUVEAU CHEMIN
                    'ec_file_path_original': row['ec_file_path'],  # Garder l'original en référence
                    'eo_file_path': None,  # Plus de fichiers EO
                    'eeg_dir': str(split_dir)  # Nouveau répertoire
                }
            
            with open(metadata_file, 'w') as f:
                json.dump(patients_metadata, f, indent=2)
        
        # Calculate totals
        total_patients = sum(len(df) for df in splits_data.values())
        
        # Save global summary
        summary = {
            'dataset_name': 'TDBRAIN',
            'total_patients': total_patients,
            'total_files_copied': total_copied,
            'condition_mapping': self.condition_mapping,
            'gender_mapping': {0: 'Female', 1: 'Male'},
            'splits': {split: len(df) for split, df in splits_data.items()},
            'demographics_summary': {}
        }
        
        # Add demographics summary for each split
        for split_name, df in splits_data.items():
            if len(df) > 0:
                summary['demographics_summary'][split_name] = {
                    'age_stats': {
                        'mean': float(df['age'].mean()), 
                        'std': float(df['age'].std())
                    },
                    'gender_distribution': {int(k): int(v) for k, v in df['gender'].value_counts().items()},
                    'condition_distribution': {str(k): int(v) for k, v in df['condition'].value_counts().items()}
                }
        
        summary_file = self.target_root / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Created dataset structure with {total_patients} patients and {total_copied} EEG files")
        for split, df in splits_data.items():
            print(f"  - {split}: {len(df)} patients")
            condition_counts = df['condition'].value_counts()
            for condition, count in condition_counts.items():
                print(f"    - {condition}: {count}")
    
    def run_reduction(self):
        """Execute complete TDBRAIN dataset reduction process using pre-selected patients"""
        # Load pre-selected dataset
        if not self.preselected_dataset:
            print("ERROR: No pre-selected dataset path provided")
            return {}, {}, {}
            
        selected_df = self.load_preselected_dataset()
        if len(selected_df) == 0:
            return {}, {}, {}
        
        # Scan EEG files
        eeg_files = self.scan_eeg_files()
        if len(eeg_files) == 0:
            return {}, {}, {}
        
        # Match selected patients with EEG files
        splits_data = self.merge_demographics_and_eeg(selected_df, eeg_files)
        if not splits_data:
            return {}, {}, {}
        
        # Create structure
        self.create_reduced_dataset_structure(splits_data)
        
        return splits_data.get('train', pd.DataFrame()), splits_data.get('val', pd.DataFrame()), splits_data.get('test', pd.DataFrame())


class EyesOpenCleaner:
    """
    Class to clean Eyes Open (EO) files from reduced dataset
    """
    
    def __init__(self, reduced_dataset_root: str):
        """
        Initialize Eyes Open cleaner
        
        Args:
            reduced_dataset_root: Path to TDBRAIN_reduced dataset
        """
        self.reduced_root = Path(reduced_dataset_root)
        self.splits = ['train', 'val', 'test']
    
    def scan_eo_files(self) -> Dict[str, List[str]]:
        """
        Scan all splits to identify Eyes Open (EO) files in the reduced dataset
        
        Returns:
            Dictionary with split names as keys and lists of EO file paths as values
        """
        eo_files_by_split = {}
        
        for split in self.splits:
            split_dir = self.reduced_root / split
            
            if not split_dir.exists():
                continue
            
            # Chercher directement les fichiers EO dans le répertoire du split
            eo_files = []
            for file_path in split_dir.glob("*_task-restEO_eeg.csv"):
                eo_files.append(str(file_path))
            
            eo_files_by_split[split] = eo_files
            print(f"  📁 {split}: {len(eo_files)} fichiers EO trouvés")
        
        return eo_files_by_split
    
    def remove_eo_files(self, eo_files_by_split: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Remove all Eyes Open (EO) files
        
        Args:
            eo_files_by_split: Dictionary of EO files by split
            
        Returns:
            Dictionary with removal statistics by split
        """
        removal_stats = {}
        
        for split, eo_files in eo_files_by_split.items():
            removed_count = 0
            
            for eo_file_path in eo_files:
                eo_file = Path(eo_file_path)
                if eo_file.exists():
                    try:
                        eo_file.unlink()  # Delete file
                        removed_count += 1
                        print(f"  🗑️ Removed: {eo_file.name}")
                    except Exception as e:
                        print(f"  ❌ Failed to remove {eo_file.name}: {e}")
            
            removal_stats[split] = removed_count
        
        return removal_stats
    
    def update_metadata_remove_eo_references(self):
        """
        Update metadata files to remove EO file references
        """
        for split in self.splits:
            split_dir = self.reduced_root / split
            metadata_file = split_dir / 'patients_metadata.json'
            
            if not metadata_file.exists():
                continue
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                patients_metadata = json.load(f)
            
            # Remove EO file references
            for patient_id, metadata in patients_metadata.items():
                metadata['eo_file_path'] = None
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(patients_metadata, f, indent=2)
    
    def verify_ec_integrity(self) -> Dict[str, Dict[str, int]]:
        """
        Verify that exactly one EC file exists per patient in the reduced dataset
        
        Returns:
            Dictionary with integrity statistics by split
        """
        integrity_stats = {}
        
        for split in self.splits:
            split_dir = self.reduced_root / split
            metadata_file = split_dir / 'patients_metadata.json'
            
            if not metadata_file.exists():
                continue
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                patients_metadata = json.load(f)
            
            total_patients = len(patients_metadata)
            valid_ec_files = 0
            missing_ec_files = 0
            
            for patient_id, metadata in patients_metadata.items():
                ec_file_path = metadata.get('ec_file_path')
                if ec_file_path and Path(ec_file_path).exists():
                    valid_ec_files += 1
                else:
                    missing_ec_files += 1
                    print(f"  ⚠️  Missing EC file for {patient_id}: {ec_file_path}")
            
            integrity_stats[split] = {
                'total_patients': total_patients,
                'valid_ec_files': valid_ec_files,
                'missing_ec_files': missing_ec_files,
                'integrity_ok': missing_ec_files == 0
            }
        
        return integrity_stats
    
    def run_cleaning(self):
        """
        Execute complete Eyes Open cleaning process
        """
        print(f"🧹 Starting Eyes Open cleaning...")
        
        # Scan EO files
        eo_files_by_split = self.scan_eo_files()
        
        # Remove EO files
        removal_stats = self.remove_eo_files(eo_files_by_split)
        
        # Update metadata
        self.update_metadata_remove_eo_references()
        
        # Verify integrity
        integrity_stats = self.verify_ec_integrity()
        
        # Print results
        total_removed = sum(removal_stats.values())
        print(f"✅ Eyes Open cleaning completed!")
        print(f"   Total EO files removed: {total_removed}")
        for split, count in removal_stats.items():
            print(f"   - {split}: {count} files removed")
        
        # Print integrity check
        print(f"📊 EC Files integrity check:")
        for split, stats in integrity_stats.items():
            status = "✅ OK" if stats['integrity_ok'] else "❌ ISSUES"
            print(f"   - {split}: {stats['valid_ec_files']}/{stats['total_patients']} EC files {status}")
        
        return removal_stats, integrity_stats


class IntelligentDownsampler:
    """
    Class to perform intelligent downsampling from 250 Hz to 90 Hz
    Creates a new dataset directory with downsampled data
    """
    
    def __init__(self, reduced_dataset_root: str, 
                 output_dataset_root: str = None,
                 target_sfreq: float = 90.0,
                 anti_alias_freq: float = 45.0):
        """
        Initialize intelligent downsampler
        
        Args:
            reduced_dataset_root: Path to TDBRAIN_reduced dataset (source)
            output_dataset_root: Path to TDBRAIN_reduced_timeseries (destination)
            target_sfreq: Target sampling frequency (90 Hz)
            anti_alias_freq: Anti-aliasing filter frequency (45 Hz)
        """
        self.reduced_root = Path(reduced_dataset_root)
        
        # Si pas de dossier de sortie spécifié, créer TDBRAIN_reduced_timeseries
        if output_dataset_root is None:
            self.output_root = self.reduced_root.parent / f"{self.reduced_root.name}_timeseries"
        else:
            self.output_root = Path(output_dataset_root)
            
        self.target_sfreq = target_sfreq
        self.anti_alias_freq = anti_alias_freq
        self.splits = ['train', 'val', 'test']
        
        print(f"📂 Source dataset: {self.reduced_root}")
        print(f"📂 Output dataset: {self.output_root}")
        
        # EEG frequency bands for validation
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }
    
    def copy_metadata_structure(self):
        """
        Copy the metadata structure from source to destination
        """
        print(f"📋 Copying metadata structure...")
        
        # Créer le dossier de destination
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Copier le dataset_summary.json s'il existe
        source_summary = self.reduced_root / 'dataset_summary.json'
        if source_summary.exists():
            dest_summary = self.output_root / 'dataset_summary.json'
            shutil.copy2(source_summary, dest_summary)
            print(f"  ✅ Copied dataset_summary.json")
        
        # Créer les dossiers splits et copier les métadonnées
        for split in self.splits:
            source_split_dir = self.reduced_root / split
            dest_split_dir = self.output_root / split
            
            if source_split_dir.exists():
                dest_split_dir.mkdir(parents=True, exist_ok=True)
                
                # Copier patients_metadata.json
                source_metadata = source_split_dir / 'patients_metadata.json'
                if source_metadata.exists():
                    dest_metadata = dest_split_dir / 'patients_metadata.json'
                    shutil.copy2(source_metadata, dest_metadata)
                    print(f"  ✅ Copied {split}/patients_metadata.json")
    
    def load_eeg_data(self, file_path: str) -> Tuple[np.ndarray, float]:
        """
        Load EEG data from CSV file
        
        Args:
            file_path: Path to EEG CSV file
            
        Returns:
            Tuple of (data array, original sampling frequency)
        """
        # Load CSV data
        data = pd.read_csv(file_path)
        
        # Convert to numpy array (assuming channels are columns)
        eeg_array = data.values.T  # Transpose to get (channels, time_points)
        
        # Assume original sampling frequency is 250 Hz for TDBRAIN
        original_sfreq = 250.0
        
        return eeg_array, original_sfreq
    
    def apply_anti_aliasing_filter(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Apply anti-aliasing low-pass filter before downsampling
        
        Args:
            data: EEG data array (channels, time_points)
            sfreq: Original sampling frequency
            
        Returns:
            Filtered data array
        """
        # Design low-pass filter at 45 Hz
        nyquist = sfreq / 2
        normalized_cutoff = self.anti_alias_freq / nyquist
        
        # Use scipy butter filter
        from scipy.signal import butter, filtfilt
        
        # 4th order Butterworth low-pass filter
        b, a = butter(4, normalized_cutoff, btype='low')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch_idx in range(data.shape[0]):
            filtered_data[ch_idx, :] = filtfilt(b, a, data[ch_idx, :])
        
        return filtered_data
    
    def downsample_data(self, data: np.ndarray, original_sfreq: float) -> np.ndarray:
        """
        Downsample data from original frequency to target frequency
        
        Args:
            data: Filtered EEG data array (channels, time_points)
            original_sfreq: Original sampling frequency
            
        Returns:
            Downsampled data array
        """
        # Calculate decimation factor
        decimation_factor = int(original_sfreq / self.target_sfreq)
        
        # Simple decimation (take every nth sample)
        downsampled_data = data[:, ::decimation_factor]
        
        return downsampled_data
    
    def validate_spectral_integrity(self, original_data: np.ndarray, 
                                  downsampled_data: np.ndarray,
                                  original_sfreq: float) -> Dict[str, Dict[str, float]]:
        """
        Validate that EEG frequency bands are preserved after downsampling
        
        Args:
            original_data: Original EEG data
            downsampled_data: Downsampled EEG data
            original_sfreq: Original sampling frequency
            
        Returns:
            Dictionary with power ratios for each frequency band
        """
        from scipy.signal import welch
        
        validation_results = {}
        
        # Take first channel for validation
        original_signal = original_data[0, :]
        downsampled_signal = downsampled_data[0, :]
        
        # Compute power spectral density for original signal
        freqs_orig, psd_orig = welch(original_signal, fs=original_sfreq, nperseg=1024)
        
        # Compute power spectral density for downsampled signal
        freqs_down, psd_down = welch(downsampled_signal, fs=self.target_sfreq, nperseg=512)
        
        # Calculate power in each frequency band
        for band_name, (low_freq, high_freq) in self.eeg_bands.items():
            # Original signal power in band
            band_mask_orig = (freqs_orig >= low_freq) & (freqs_orig <= high_freq)
            power_orig = np.trapz(psd_orig[band_mask_orig], freqs_orig[band_mask_orig])
            
            # Downsampled signal power in band (if frequencies are available)
            band_mask_down = (freqs_down >= low_freq) & (freqs_down <= high_freq)
            if np.any(band_mask_down):
                power_down = np.trapz(psd_down[band_mask_down], freqs_down[band_mask_down])
                power_ratio = power_down / power_orig if power_orig > 0 else 0
            else:
                power_ratio = 0
            
            validation_results[band_name] = {
                'power_original': power_orig,
                'power_downsampled': power_down if np.any(band_mask_down) else 0,
                'power_ratio': power_ratio
            }
        
        return validation_results
    
    def process_single_file(self, file_path: str, output_dir: Path) -> Tuple[str, Dict]:
        """
        Process a single EEG file with downsampling and save to new location
        
        Args:
            file_path: Path to original EEG CSV file
            output_dir: Directory to save downsampled file
            
        Returns:
            Tuple of (output_path, validation_results)
        """
        # Load original data
        original_data, original_sfreq = self.load_eeg_data(file_path)
        
        # Apply anti-aliasing filter
        filtered_data = self.apply_anti_aliasing_filter(original_data, original_sfreq)
        
        # Downsample
        downsampled_data = self.downsample_data(filtered_data, original_sfreq)
        
        # Validate spectral integrity
        validation_results = self.validate_spectral_integrity(
            original_data, downsampled_data, original_sfreq
        )
        
        # Generate output filename in new directory
        file_path_obj = Path(file_path)
        output_filename = f"{file_path_obj.stem}_90Hz.csv"
        output_path = output_dir / output_filename
        
        # Créer le répertoire de sortie si nécessaire
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save downsampled data
        downsampled_df = pd.DataFrame(downsampled_data.T)  # Transpose back to (time, channels)
        downsampled_df.to_csv(output_path, index=False)
        
        return str(output_path), validation_results
    
    def update_metadata_with_downsampled_paths(self):
        """
        Update metadata files in the NEW directory to reference downsampled files
        """
        for split in self.splits:
            source_split_dir = self.reduced_root / split
            dest_split_dir = self.output_root / split
            metadata_file = dest_split_dir / 'patients_metadata.json'
            
            if not metadata_file.exists():
                continue
            
            # Load metadata from destination
            with open(metadata_file, 'r') as f:
                patients_metadata = json.load(f)
            
            # Update EC file paths to point to downsampled versions in new location
            for patient_id, metadata in patients_metadata.items():
                original_ec_path = metadata.get('ec_file_path')
                if original_ec_path:
                    # Créer le nouveau chemin dans le dossier timeseries
                    original_path_obj = Path(original_ec_path)
                    downsampled_filename = f"{original_path_obj.stem}_90Hz.csv"
                    
                    # Le nouveau chemin sera dans dest_split_dir
                    new_ec_path = str(dest_split_dir / downsampled_filename)
                    
                    # Mettre à jour les métadonnées
                    metadata['ec_file_path_original'] = original_ec_path  # Garder l'original
                    metadata['ec_file_path'] = new_ec_path  # Nouveau chemin downsampld
                    metadata['original_sfreq'] = 250.0
                    metadata['downsampled_sfreq'] = self.target_sfreq
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(patients_metadata, f, indent=2)
            
            print(f"  ✅ Updated metadata for {split} split")
    
    def run_downsampling(self):
        """
        Execute complete intelligent downsampling process
        Creates a new timeseries dataset directory
        """
        print(f"🎯 Starting intelligent downsampling...")
        print(f"   Source: {self.reduced_root}")
        print(f"   Destination: {self.output_root}")
        
        # Étape 1: Copier la structure des métadonnées
        self.copy_metadata_structure()
        
        # Étape 2: Traiter les fichiers EEG
        all_validation_results = {}
        
        for split in self.splits:
            source_split_dir = self.reduced_root / split
            dest_split_dir = self.output_root / split
            metadata_file = source_split_dir / 'patients_metadata.json'
            
            if not metadata_file.exists():
                print(f"⚠️  No metadata found for {split}, skipping...")
                continue
            
            print(f"🔄 Processing {split} split...")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                patients_metadata = json.load(f)
            
            split_validation = {}
            processed_count = 0
            
            # Process each patient's EC file
            for patient_id, metadata in patients_metadata.items():
                ec_file_path = metadata.get('ec_file_path')
                if ec_file_path and Path(ec_file_path).exists():
                    try:
                        output_path, validation_results = self.process_single_file(
                            ec_file_path, dest_split_dir
                        )
                        split_validation[patient_id] = validation_results
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            print(f"   Processed {processed_count} files...")
                            
                    except Exception as e:
                        print(f"   ❌ Error processing {patient_id}: {e}")
                else:
                    print(f"   ⚠️  EEG file not found for {patient_id}: {ec_file_path}")
            
            all_validation_results[split] = split_validation
            print(f"   ✅ {split}: {processed_count} files processed")
        
        # Étape 3: Mettre à jour les métadonnées avec les nouveaux chemins
        print(f"📝 Updating metadata with new file paths...")
        self.update_metadata_with_downsampled_paths()
        
        # Étape 4: Sauvegarder les résultats de validation
        validation_file = self.output_root / 'downsampling_validation.json'
        with open(validation_file, 'w') as f:
            json.dump(all_validation_results, f, indent=2)
        
        print(f"💾 Validation results saved to: {validation_file}")
        
        # Résumé final
        total_processed = sum(len(split_val) for split_val in all_validation_results.values())
        print(f"\n🎉 Downsampling completed!")
        print(f"   📂 New dataset created: {self.output_root}")
        print(f"   📊 Total files processed: {total_processed}")
        print(f"   🔄 Frequency: 250Hz → {self.target_sfreq}Hz")
        
        return all_validation_results


class FrequencyBandTokenizer:
    """
    NEW STEP 4: Class to create frequency band tokens from downsampled EEG signals
    Following the original XAIguiFormer approach for frequency band processing
    """
    
    def __init__(self, 
                 timeseries_dataset_root: str,
                 target_sfreq: float = 90.0):
        """
        Initialize frequency band tokenizer
        
        Args:
            timeseries_dataset_root: Path to TDBRAIN_reduced_timeseries dataset
            target_sfreq: Sampling frequency of the input data (90 Hz)
        """
        self.timeseries_root = Path(timeseries_dataset_root)
        self.target_sfreq = target_sfreq
        self.splits = ['train', 'val', 'test']
        
        # Frequency bands from original XAIguiFormer (TDBRAIN_preprocess.yaml)
        self.frequency_bands = {
            'delta': [2., 4.],
            'theta': [4., 8.],
            'low_alpha': [8., 10.],
            'high_alpha': [10., 12.],
            'low_beta': [12., 18.],
            'mid_beta': [18., 21.],
            'high_beta': [21., 30.],
            'gamma': [30., 45.],
            'beta': [12., 30.]  # For theta/beta ratio
        }
        
        print(f"📂 Timeseries dataset: {self.timeseries_root}")
        print(f"🎵 Processing {len(self.frequency_bands)} frequency bands")
        
    def create_bandpass_filter(self, low_freq: float, high_freq: float, 
                             filter_order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a Butterworth bandpass filter for specific frequency band
        
        Args:
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            filter_order: Filter order (default: 4)
            
        Returns:
            Tuple of filter coefficients (b, a)
        """
        nyquist = self.target_sfreq / 2
        low_normalized = low_freq / nyquist
        high_normalized = high_freq / nyquist
        
        # Ensure frequencies are within valid range
        low_normalized = max(0.01, min(0.99, low_normalized))
        high_normalized = max(0.01, min(0.99, high_normalized))
        
        if low_normalized >= high_normalized:
            high_normalized = min(0.99, low_normalized + 0.01)
        
        b, a = butter(filter_order, [low_normalized, high_normalized], btype='band')
        return b, a
    
    def apply_frequency_filter(self, data: np.ndarray, band_name: str) -> np.ndarray:
        """
        Apply frequency band filter to EEG data
        
        Args:
            data: EEG data array (channels, time_points)
            band_name: Name of frequency band
            
        Returns:
            Filtered data array
        """
        low_freq, high_freq = self.frequency_bands[band_name]
        
        # Handle special case for gamma band (might exceed Nyquist at 90Hz)
        if high_freq > (self.target_sfreq / 2):
            print(f"⚠️  Warning: {band_name} band upper limit ({high_freq}Hz) exceeds Nyquist ({self.target_sfreq/2}Hz)")
            high_freq = min(high_freq, self.target_sfreq / 2 - 1)
        
        # Create and apply bandpass filter
        b, a = self.create_bandpass_filter(low_freq, high_freq)
        
        filtered_data = np.zeros_like(data)
        for ch_idx in range(data.shape[0]):
            filtered_data[ch_idx, :] = filtfilt(b, a, data[ch_idx, :])
        
        return filtered_data
    
    def process_single_patient(self, eeg_file_path: str, patient_id: str, 
                             output_dir: Path) -> str:
        """
        Process a single patient's EEG file to create frequency band tokens
        
        Args:
            eeg_file_path: Path to patient's downsampled EEG CSV file
            patient_id: Patient identifier
            output_dir: Directory to save frequency tokens
            
        Returns:
            Path to saved frequency tokens file
        """
        # Load downsampled EEG data
        eeg_df = pd.read_csv(eeg_file_path)
        eeg_data = eeg_df.values.T  # (channels, time_points)
        
        # Initialize frequency tokens array: (num_bands, channels, time_points)
        num_bands = len(self.frequency_bands)
        num_channels, num_timepoints = eeg_data.shape
        frequency_tokens = np.zeros((num_bands, num_channels, num_timepoints))
        
        # Apply each frequency band filter
        for band_idx, (band_name, _) in enumerate(self.frequency_bands.items()):
            try:
                filtered_data = self.apply_frequency_filter(eeg_data, band_name)
                frequency_tokens[band_idx, :, :] = filtered_data
                
            except Exception as e:
                print(f"  ⚠️  Error filtering {band_name} for {patient_id}: {e}")
                # Fill with zeros if filtering fails
                frequency_tokens[band_idx, :, :] = np.zeros((num_channels, num_timepoints))
        
        # Create frequency tokens directory
        tokens_dir = output_dir / 'frequency_tokens'
        tokens_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frequency tokens (Option B: single file per patient)
        tokens_filename = f"{patient_id}_frequency_bands.npy"
        tokens_path = tokens_dir / tokens_filename
        
        np.save(tokens_path, frequency_tokens)
        
        return str(tokens_path)
    
    def update_metadata_with_tokens(self):
        """
        Update metadata files to reference frequency tokens
        """
        for split in self.splits:
            split_dir = self.timeseries_root / split
            metadata_file = split_dir / 'patients_metadata.json'
            
            if not metadata_file.exists():
                continue
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                patients_metadata = json.load(f)
            
            # Add frequency tokens paths
            for patient_id, metadata in patients_metadata.items():
                tokens_path = split_dir / 'frequency_tokens' / f"{patient_id}_frequency_bands.npy"
                metadata['frequency_tokens_path'] = str(tokens_path)
                metadata['frequency_bands'] = list(self.frequency_bands.keys())
                metadata['num_frequency_bands'] = len(self.frequency_bands)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(patients_metadata, f, indent=2)
            
            print(f"  ✅ Updated metadata with tokens for {split} split")
    
    def run_tokenization(self):
        """
        Execute complete frequency band tokenization process
        """
        print(f"🎼 Starting frequency band tokenization...")
        
        all_processing_results = {}
        
        for split in self.splits:
            split_dir = self.timeseries_root / split
            metadata_file = split_dir / 'patients_metadata.json'
            
            if not metadata_file.exists():
                print(f"⚠️  No metadata found for {split}, skipping...")
                continue
            
            print(f"🔄 Processing {split} split...")
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                patients_metadata = json.load(f)
            
            split_results = {}
            processed_count = 0
            
            # Process each patient's downsampled EEG file
            for patient_id, metadata in patients_metadata.items():
                eeg_file_path = metadata.get('ec_file_path')
                if eeg_file_path and Path(eeg_file_path).exists():
                    try:
                        tokens_path = self.process_single_patient(
                            eeg_file_path, patient_id, split_dir
                        )
                        split_results[patient_id] = {
                            'tokens_path': tokens_path,
                            'success': True
                        }
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            print(f"   Processed {processed_count} files...")
                            
                    except Exception as e:
                        print(f"   ❌ Error processing {patient_id}: {e}")
                        split_results[patient_id] = {
                            'tokens_path': None,
                            'success': False,
                            'error': str(e)
                        }
                else:
                    print(f"   ⚠️  EEG file not found for {patient_id}: {eeg_file_path}")
            
            all_processing_results[split] = split_results
            print(f"   ✅ {split}: {processed_count} files processed")
        
        # Update metadata with token paths
        print(f"📝 Updating metadata with frequency tokens...")
        self.update_metadata_with_tokens()
        
        # Save processing results
        results_file = self.timeseries_root / 'frequency_tokenization_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_processing_results, f, indent=2)
        
        print(f"💾 Tokenization results saved to: {results_file}")
        
        # Final summary
        total_processed = sum(
            len([r for r in split_results.values() if r['success']]) 
            for split_results in all_processing_results.values()
        )
        print(f"\n🎉 Frequency band tokenization completed!")
        print(f"   📂 Tokens saved in: {self.timeseries_root}")
        print(f"   📊 Total patients processed: {total_processed}")
        print(f"   🎵 Frequency bands: {len(self.frequency_bands)} bands per patient")
        
        return all_processing_results


def test_frequency_tokens():
    """
    Test function to verify frequency tokens dimensions and content
    """
    print("🧪 Testing frequency tokens...")
    
    # Get project root path
    project_root = Path(__file__).parent.parent
    timeseries_root = project_root / "data" / "TDBRAIN_reduced_timeseries"
    
    # Test on train split
    train_dir = timeseries_root / "train"
    metadata_file = train_dir / 'patients_metadata.json'
    
    if not metadata_file.exists():
        print("❌ No metadata file found for testing")
        return False
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        patients_metadata = json.load(f)
    
    # Test first patient
    first_patient_id = list(patients_metadata.keys())[0]
    first_patient_data = patients_metadata[first_patient_id]
    
    tokens_path = first_patient_data.get('frequency_tokens_path')
    if not tokens_path or not Path(tokens_path).exists():
        print(f"❌ Tokens file not found: {tokens_path}")
        return False
    
    # Load frequency tokens
    try:
        frequency_tokens = np.load(tokens_path)
        print(f"✅ Successfully loaded tokens for {first_patient_id}")
        print(f"   Shape: {frequency_tokens.shape}")
        print(f"   Expected: (9, 33, ~time_points)")
        print(f"   Frequency bands: {frequency_tokens.shape[0]}")
        print(f"   Channels: {frequency_tokens.shape[1]}")
        print(f"   Time points: {frequency_tokens.shape[2]}")
        
        # Verify expected dimensions
        expected_bands = 9
        expected_channels = 33
        
        if frequency_tokens.shape[0] != expected_bands:
            print(f"⚠️  Unexpected number of frequency bands: {frequency_tokens.shape[0]} (expected {expected_bands})")
        
        if frequency_tokens.shape[1] != expected_channels:
            print(f"⚠️  Unexpected number of channels: {frequency_tokens.shape[1]} (expected {expected_channels})")
        
        # Check for non-zero values (sanity check)
        non_zero_bands = np.count_nonzero(frequency_tokens, axis=(1, 2))
        print(f"   Non-zero values per band: {non_zero_bands}")
        
        # Test multiple patients
        test_count = min(3, len(patients_metadata))
        all_shapes = []
        
        for i, (patient_id, patient_data) in enumerate(list(patients_metadata.items())[:test_count]):
            tokens_path = patient_data.get('frequency_tokens_path')
            if tokens_path and Path(tokens_path).exists():
                tokens = np.load(tokens_path)
                all_shapes.append(tokens.shape)
                print(f"   Patient {i+1} ({patient_id}): {tokens.shape}")
        
        # Check shape consistency
        if len(set(all_shapes)) == 1:
            print(f"✅ All tested patients have consistent shapes")
        else:
            print(f"⚠️  Inconsistent shapes across patients: {set(all_shapes)}")
        
        print(f"🎉 Frequency tokens test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading tokens: {e}")
        return False


def main():
    """Main function to execute complete preprocessing pipeline including Step 4"""
    import sys
    
    # Configuration pour projet structure (script est dans utils/, data dans data/)
    project_root = Path(__file__).parent.parent  # Remonte de utils/ vers la racine du projet
    tdbrain_root = project_root / "data" / "TDBRAIN-dataset-derivatives"
    participants_csv = project_root / "data" / "TDBRAIN_participants_V2.csv"
    preselected_dataset = project_root / "data" / "TDBRAIN_90_patients.csv"
    target_root = project_root / "data" / "TDBRAIN_reduced"
    timeseries_root = project_root / "data" / "TDBRAIN_reduced_timeseries"
    
    print(f"Project root: {project_root}")
    print(f"TDBRAIN root: {tdbrain_root}")
    print(f"Participants CSV: {participants_csv}")
    print(f"Pre-selected dataset: {preselected_dataset}")
    print(f"Target root: {target_root}")
    print(f"Timeseries root: {timeseries_root}")
    print(f"TDBRAIN root exists: {tdbrain_root.exists()}")
    print(f"Participants CSV exists: {participants_csv.exists()}")
    print(f"Pre-selected dataset exists: {preselected_dataset.exists()}")
    
    # Check command line arguments for step selection
    steps_to_run = sys.argv[1:] if len(sys.argv) > 1 else ['all']
    print(f"Steps to run: {steps_to_run}")
    
    if 'step1' in steps_to_run or 'all' in steps_to_run:
        print("=== STARTING STEP 1: Dataset Reduction ===")
        # Step 1: Dataset reduction using pre-selected patients
        reducer = TDBRAINDatasetReducer(
            tdbrain_root=str(tdbrain_root),
            participants_csv=str(participants_csv),
            target_root=str(target_root),
            preselected_dataset=str(preselected_dataset)  # NOUVEAU
        )
        
        train_df, val_df, test_df = reducer.run_reduction()
        
        if len(train_df) > 0:
            print(f"SUCCESS: Created dataset with {len(train_df)} train, {len(val_df)} val, {len(test_df)} test patients")
        else:
            print("FAILED: No patients in final dataset")
    
    if 'step2' in steps_to_run or 'all' in steps_to_run:
        print("=== STARTING STEP 2: Eyes Open Cleaning ===")
        # Step 2: Eyes Open cleaning (keep only Eyes Closed recordings)
        cleaner = EyesOpenCleaner(str(target_root))
        removal_stats, integrity_stats = cleaner.run_cleaning()
    
    if 'step3' in steps_to_run or 'all' in steps_to_run:
        print("=== STARTING STEP 3: Intelligent Downsampling ===")
        # Step 3: Intelligent downsampling (250 Hz -> 90 Hz avec nouveau dossier)
        downsampler = IntelligentDownsampler(
            reduced_dataset_root=str(target_root),
            output_dataset_root=str(timeseries_root)
        )
        validation_results = downsampler.run_downsampling()
    
    if 'step4' in steps_to_run or 'all' in steps_to_run:
        print("=== STARTING STEP 4: Frequency Band Tokenization ===")
        # Step 4: NEW - Create frequency band tokens from downsampled EEG
        tokenizer = FrequencyBandTokenizer(
            timeseries_dataset_root=str(timeseries_root)
        )
        tokenization_results = tokenizer.run_tokenization()
    
    if 'test' in steps_to_run:
        print("=== RUNNING TESTS ===")
        # Test frequency tokens
        test_frequency_tokens()
    
    print("=== PREPROCESSING COMPLETE ===")
    print("Data ready for MultiROCKET tokenization in model!")


if __name__ == "__main__":
    main()