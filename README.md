# XAIguiFormer – Reproducibility Challenge
> Reproduction of: **XAIGUIFORMER: Explainable Artificial Intelligence Guided Transformer for Brain Disorder Identification** (ICLR 2025)  
> ML Reproducibility Challenge – Branch main

---

## IMPORTANT: This github contains 2 branchs:
> branch main --> Reproducibility Challenge
> branch main2_XaiGuiFormerTimeSeries --> New approach with TimeSeries model (see README inside)

---

##  Project Overview

This repository contains our implementation of **XAIguiFormer** for the ML Reproducibility Challenge, based on the ICLR 2025 paper "XAIguiFormer: Explainable Artificial Intelligence Guided Transformer for Brain Disorder Identification" by Guo et al.

XAIguiFormer introduces a novel architecture that leverages explainable AI (XAI) not just for interpretability, but as an active component to enhance transformer performance in EEG-based brain disorder classification through connectome analysis.

---

##  Key Innovation

**XAIguiFormer** revolutionizes the use of XAI by employing explanations to directly refine the self-attention mechanism, moving beyond traditional post-hoc interpretability to performance enhancement through:
- **Dual-pass architecture**: Standard transformer followed by XAI-guided refinement
- **Real-time explanation integration**: DeepLIFT explanations refine Query/Key matrices during training
- **Connectome preservation**: Atomic tokenization maintains graph topological structure

---

## Core Contributions

### 1.  **Connectome Tokenizer**
- Treats single-band graphs as atomic tokens to preserve topological structure
- Generates frequency sequences without fragmenting connectivity patterns
- GNN-based encoding with GINEConv for inductive bias on small datasets

### 2.  **dRoFE (demographic Rotary Frequency Encoding)**
- Integrates frequency bounds and demographic information (age, gender) into token embeddings
- Rotary matrix-based encoding inspired by RoPE but adapted for EEG characteristics
- Mitigates individual differences while preserving frequency-specific information

### 3.  **XAI-Guided Self-Attention**
- Concurrent explanation generation using multi-layer DeepLIFT
- Refined Query/Key matrices based on feature importance scores
- Dynamical system-inspired approach for adaptive attention refinement

### 4.  **Dual Loss Function**
- Combined coarse and refined predictions with configurable weighting (α)
- Enables end-to-end training with XAI guidance supervision

---

##  Architecture Overview

```
EEG Signals → Multi-band Connectomes → GNN Tokenizer → Transformer Encoder
                                                            ↓
                                     XAI Explainer ← Standard Predictions
                                            ↓
                                   Refined Q/K Matrices
                                            ↓
                              XAI-Guided Transformer → Final Predictions
```

---

## Poetry: Dependency Management

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and environments.
Poetry is a modern alternative to pip + virtualenv + requirements.txt, offering better reproducibility and project isolation.

###  Install Poetry
If Poetry is not installed yet:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Then check:
```bash
poetry --version
```
If that fails, follow the guide to configure your shell:  
 https://python-poetry.org/docs/#installation

###  Using Poetry in this Project
All commands below must be run from the root of the project (where `pyproject.toml` is located).

#### 1. Install all dependencies
```bash
poetry install
```
This will:
- Create a virtual environment (if not already created)
- Install all required packages from `pyproject.toml`

#### 2. Activate the virtual environment
```bash
poetry shell
```
Now you can run any training or evaluation script inside the isolated environment.

Example:
```bash
python scripts/train.py
```

#### 3. Add a new dependency
```bash
poetry add <package-name>
```
Example:
```bash
poetry add tensorboard
```
This updates `pyproject.toml` automatically.

#### 4. Export a `requirements.txt` file (optional)
For compatibility with Colab or Docker:
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

---

##  Directory Structure

```text
XAIguiFormer/
├── configs/              # YAML configs for training & experiments
│   ├── base.yaml
│   └── tdbrain_config.yaml
├── data/                 # EEG datasets and preprocessing
│   └── TDBRAIN/
│       ├── raw/          # Raw EEG data in BIDS format
│       ├── preprocessed/ # PREP pipeline processed EEG epochs
│       ├── connectome/   # Multi-frequency band connectivity matrices
│       └── tokens/       # PyTorch Geometric graph objects
├── src/                  # Core implementation
│   ├── model/            # Model components
│   └── preprocessing/    # EEG processing pipeline
├── scripts/              # Training and evaluation scripts
├── checkpoints/          # Trained model weights
├── results/              # Evaluation outputs and visualizations
├── pyproject.toml        # Poetry-managed dependencies
└── README.md
```

###  Configuration (`configs/`)
- **`base.yaml`**: Basic model architecture parameters and training settings
- **`tdbrain_config.yaml`**: TDBRAIN dataset-specific configuration including preprocessing parameters, frequency bands, and connectome construction methods

###  Data Pipeline (`data/TDBRAIN/`)
- **`raw/`**: Raw EEG data in BIDS format
- **`preprocessed/`**: PREP pipeline processed EEG epochs
- **`connectome/`**: Multi-frequency band connectivity matrices (COH + wPLI)
- **`tokens/`**: PyTorch Geometric graph objects ready for training

###  Core Implementation (`src/`)

#### Model Components (`src/model/`)
- **`xaiguiformer_pipeline.py`**: Main model architecture integrating all components
- **`transformer_used_two_times.py`**: Shared transformer encoder with dual-pass capability
- **`explainers.py`**: Multi-layer DeepLIFT explainer with Captum integration
- **`dRoFE_embedding.py`**: Demographic rotary frequency encoding implementation
- **`gnn.py`**: GINEConv-based connectome encoder with torch-scatter fallback
- **`losses.py`**: XAI-guided loss function with class weight balancing

#### Preprocessing Pipeline (`src/preprocessing/`)
- **`eeg_preprocessing.py`**: Complete EEG preprocessing (PREP, filtering, ICA, epoching)
- **`connectome_builder.py`**: Multi-band connectivity matrix construction
- **`aggregate_tokens.py`**: Graph aggregation into unified PyG batches

#### Training Scripts (`scripts/`)
- **`train.py`**: Main training loop with adaptive dataset splitting
- **`evaluate.py`**: Comprehensive evaluation with balanced accuracy focus
- **`plot_attention_maps.py`**: Attention visualization and interpretability analysis

---

##  Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM for TDBRAIN dataset processing

### Environment Setup

Using Poetry (recommended):
```bash
git clone <repository-url>
cd XAIguiFormer
poetry install
poetry shell
```

Alternative using pip:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install mne mne-connectivity mne-icalabel pyprep
pip install captum yacs scikit-learn
pip install pandas numpy matplotlib seaborn tqdm
```

---

##  Usage

### 1.  Data Preparation

**Configure preprocessing parameters:**
```bash
# Edit configs/tdbrain_config.yaml for your dataset paths
vim configs/tdbrain_config.yaml
```

**Run preprocessing pipeline:**
```bash
# Step 1: EEG preprocessing (PREP, filtering, ICA)
python src/preprocessing/eeg_preprocessing.py

# Step 2: Connectome construction (COH + wPLI)
python src/preprocessing/connectome_builder.py

# Step 3: Graph aggregation
python src/preprocessing/aggregate_tokens.py
```

### 2.  Training

**Run full training with XAI guidance:**
```bash
python scripts/train.py
```

**Key training features:**
- Adaptive dataset splitting for small datasets
- Concurrent XAI explanation generation
- Dual-pass architecture with shared weights
- Automatic model checkpointing

### 3. 📈 Evaluation

**Evaluate trained model:**
```bash
python scripts/evaluate.py
```

**Generate attention visualizations:**
```bash
python scripts/plot_attention_maps.py
```

---

##  Technical Implementation Details

###  XAI-Guided Architecture
- **Concurrent Processing**: XAI explanations generated in parallel with standard forward pass
- **Multi-layer Explanations**: DeepLIFT applied to each transformer layer
- **Captum Integration**: Wrapper classes for explanation compatibility
- **Fallback Mechanisms**: Graceful degradation when XAI components fail

###  Memory Optimization
- **Torch-scatter Fallback**: Custom implementation when dependency unavailable
- **Gradient Clipping**: Prevents exploding gradients during XAI refinement
- **Efficient Batching**: Smart batch size adaptation for small datasets

###  Reproducibility Features
- **Fixed Random Seeds**: Consistent results across runs
- **Deterministic Operations**: CUDA deterministic mode support
- **Configuration Tracking**: Complete hyperparameter logging

---

##  Datasets

###  TDBRAIN (Primary)
- **Size**: 88 patients across 4 diagnostic categories
- **Modality**: Eyes-closed resting-state EEG (26 channels)
- **Classes**: ADHD, MDD, OCD, Healthy Controls
- **Challenge**: Small dataset requiring careful splitting and regularization

###  TUAB (Referenced)
- **Size**: 2,993 sessions (1,385 normal, 998 abnormal)
- **Modality**: Clinical EEG (19 channels)
- **Classes**: Binary classification (normal/abnormal)

---

##  Results and Evaluation

###  Key Metrics
- **Balanced Accuracy (BAC)**: Primary metric for class-imbalanced datasets
- **AUROC/AUC-PR**: Comprehensive performance assessment
- **Attention Entropy**: Quantitative measure of attention concentration

###  Expected Performance
- **TDBRAIN**: ~66% BAC with XAI guidance vs ~63% without
- **Attention Concentration**: Lower entropy indicating focused attention patterns
- **Frequency Importance**: Theta/beta ratio as primary biomarker

---

##  Reproducibility Notes

###  Known Issues
- **Torch-scatter Dependency**: Fallback implementation may have performance differences
- **Small Dataset Challenges**: TDBRAIN requires careful hyperparameter tuning
- **Captum Compatibility**: Some explanation methods may require specific PyTorch versions

###  Configuration Sensitivity
- **Alpha Parameter**: XAI guidance weight (α=0.7 recommended)
- **Learning Rate**: Dataset-specific tuning required
- **Batch Size**: Adaptive sizing for small datasets

---

##  Future Directions

###  Immediate Improvements
- **Larger Dataset Evaluation**: Validation on TUAB and other clinical datasets
- **Foundation Model Integration**: Pre-training strategies for better initialization
- **Enhanced XAI Methods**: Integration of additional explanation algorithms

###  Long-term Extensions
- **Multi-modal Integration**: Combination with other neuroimaging modalities
- **Real-time Applications**: Optimization for clinical deployment
- **Federated Learning**: Privacy-preserving multi-site training

