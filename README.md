# XAIguiFormer TimeSeries: Temporal-Preserving Variant for EEG Classification

**Branch: main2** - This branch contains our novel time-series implementation that preserves temporal dynamics in EEG signals.

## Overview

This repository implements **XAIguiFormer_TimeSeries**, an extended variant of the original XAIguiFormer architecture designed to address information compression limitations in connectome-based EEG preprocessing. While the original approach compresses EEG signals by approximately 1,350:1 through static connectivity matrices, our time-series variant preserves temporal dynamics using MultiROCKET-based tokenization.

## Key Innovation

**XAIguiFormer_TimeSeries** replaces the connectome tokenizer with a MultiROCKET-based temporal encoder while maintaining the core XAI-guided transformer architecture, enabling direct processing of frequency-filtered EEG signals without destructive temporal averaging.

## Core Contributions

- **Temporal Preservation**: Direct processing of EEG time series without connectome conversion
- **MultiROCKET Integration**: Memory-efficient temporal feature extraction with learnable channel attention
- **Architectural Compatibility**: Drop-in replacement preserving all downstream XAI components
- **Chunking Strategy**: Overlapping temporal windows for 3.2x data augmentation

### Core Components

#### Configuration Files (`configs/`)
- **`TDBRAIN_model.yaml`**: Model architecture parameters, training hyperparameters, and evaluation settings for the TDBRAIN dataset
- **`TDBRAIN_preprocess.yaml`**: EEG preprocessing pipeline configuration including filtering parameters, ICA settings, and frequency band definitions

#### Data Management (`data/`)
- **`TDBRAIN_participants_V2_data/`**: Folder containing raw patient metadata files with demographic information (age, gender) and psychiatric disorder labels
- **`TDBRAIN_participants_V2.json/tsv/xlsx`**: Patient metadata in various formats containing demographic variables and diagnostic labels for psychiatric conditions (ADHD, MDD, SMC, HEALTHY)
- **`TDBRAIN_replication_template_V2.xlsx`**: Template file for data replication studies
- **`convert_xls_to_csv.py`**: Utility to convert TDBRAIN participant metadata from Excel to CSV format for easier processing
- **`create_tdbrain_dataset_90.py`**: Creates a balanced subset of 88 patients (22 per diagnostic category) with stratified train/validation/test splits
- **`TDBRAIN_reduced_timeseries/`**: Contains preprocessed EEG time series data with frequency band tokenization, but only for a reduced dataset of 88 patients

#### Model Architecture (`models/`)
- **`xaiguiformer_timeseries.py`**: Main implementation of XAIguiFormer_TimeSeries, integrating MultiROCKET tokenization with the original XAI-guided transformer architecture

#### Model Components (`modules/`)
- **`multirocket_tokenizer.py`**: MultiROCKET-based temporal feature extractor with learnable channel attention, replacing the original connectome encoder
- **`transformer.py`**: XAI-guided transformer encoder with demographic-aware rotary frequency encoding (dRoFE)
- **`explainer.py`**: Unified interface for various XAI methods (DeepLIFT, GradCAM, Integrated Gradients)
- **`explainer_wrapper.py`**: Low-level implementations of different explainability algorithms
- **`positional_encoding_wrapper.py`**: Implementation of dRoFE for integrating demographic information into positional encodings

#### Utilities (`utils/`)
- **`preprocessing_timeseries.py`**: Complete EEG preprocessing pipeline from raw signals to frequency band tokens, including intelligent downsampling and artifact removal
- **`data_transformer_tensor_timeseries.py`**: PyTorch dataset classes and data loaders for frequency-tokenized EEG data with uniform temporal truncation
- **`eval_metrics.py`**: Evaluation metrics including balanced accuracy, sensitivity, AUROC, and AUC-PR for psychiatric classification tasks

#### Training (`main_timeseries.py`)
Main training script implementing the chunking strategy for temporal data augmentation, enabling efficient processing of long EEG sequences while preserving temporal information.

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for full dataset processing

### Environment Setup

Using Poetry (recommended):
```bash
git clone https://github.com/your-repo/XAIguiFormer
cd XAIguiFormer
poetry install
poetry shell
Alternative using pip:
bashpip install torch torchvision torchaudio
pip install torch-geometric
pip install mne mne-connectivity mne-icalabel
pip install captum timm einops yacs
pip install pandas numpy scikit-learn matplotlib
Usage
Data Preparation

Convert metadata (if needed):

bashcd data/
python convert_xls_to_csv.py

Create balanced dataset:

bashcd data/
python create_tdbrain_dataset_90.py

Preprocess EEG signals:

bashpython utils/preprocessing_timeseries.py
Training
Run the time-series model with chunking strategy:
bashpython main_timeseries.py
Technical Approach
XAIguiFormer_TimeSeries Architecture
XAIguiFormer_TimeSeries addresses information compression limitations:

MultiROCKET Tokenization: 200 random convolutional kernels with learnable channel attention
Temporal Preservation: Direct processing of frequency-filtered EEG signals without connectome conversion
Chunking Strategy: Overlapping temporal windows for data augmentation (3.2x increase in training samples)
Architectural Compatibility: Drop-in replacement maintaining all downstream XAI components

Key Technical Features

Information Preservation: Avoids 1,350:1 compression ratio of original connectome pipeline
Memory-Efficient Design: Optimized for standard GPU resources (2GB RAM usage)
Reproducible Kernels: Fixed random seeds ensure consistent MultiROCKET features
End-to-End Learning: Trainable attention mechanisms for inter-channel dependencies

Results and Limitations
Achievements

Development of computationally efficient time-series alternative to connectome processing
Comprehensive framework for EEG temporal modeling with preserved dynamics
Memory-optimized implementation compatible with standard hardware

Technical Challenges

Implementation Issues: Demographic tensor integration requires debugging for NaN resolution
Computational Constraints: Full dataset processing requires high-performance computing resources
Dependency Conflicts: Captum library compatibility issues in certain environments

Future Directions

Integration with foundation models (MANTIS, TimesNet) for enhanced temporal modeling
Native time-series transformer architectures with embedded XAI mechanisms
Extended evaluation on larger clinical datasets