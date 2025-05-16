# XAIguiFormer – Reproducibility Challenge

> Reproduction of: **XAIGUIFORMER: Explainable Artificial Intelligence Guided Transformer for Brain Disorder Identification** (ICLR 2025)  
> ML Reproducibility Challenge – Télécom Paris

---

## 📦 0. Poetry: Dependency Management

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and environments.

Poetry is a modern alternative to pip + virtualenv + requirements.txt, offering better reproducibility and project isolation.

### 🔧 Install Poetry

If Poetry is not installed yet:

```bash
curl -sSL https://install.python-poetry.org | python3 -
Then check:

bash
Copier
Modifier
poetry --version
If that fails, follow the guide to configure your shell:
👉 https://python-poetry.org/docs/#installation

▶️ Using Poetry in this Project
All commands below must be run from the root of the project (where pyproject.toml is located).

1. Install all dependencies
bash
Copier
Modifier
poetry install
This will:

Create a virtual environment (if not already created)

Install all required packages from pyproject.toml

2. Activate the virtual environment
bash
Copier
Modifier
poetry shell
Now you can run any training or evaluation script inside the isolated environment.

Example:

bash
Copier
Modifier
python scripts/train.py --config configs/tuab_config.yaml
3. Add a new dependency
bash
Copier
Modifier
poetry add <package-name>
Example:

bash
Copier
Modifier
poetry add tensorboard
This updates pyproject.toml automatically.

4. Export a requirements.txt file (optional)
For compatibility with Colab or Docker:

bash
Copier
Modifier
poetry export -f requirements.txt --output requirements.txt --without-hashes
📖 1. Project Overview
This repository contains a full reproducibility implementation of XAIguiFormer, a Transformer-based model for brain disorder identification from EEG data using XAI-guided attention refinement.

The goal is to faithfully reproduce the experimental setup, architecture, and evaluation metrics from the original paper.

🗂️ 2. Project Structure
text
Copier
Modifier
XAIguiFormer_MLReproducibility/
├── configs/              # YAML configs for training & ablation
├── data/                 # Preprocessing scripts and raw EEG folders
├── src/                  # All model components and training logic
├── scripts/              # Run scripts (train, evaluate, ablation, visualize)
├── checkpoints/          # Trained models
├── results/              # Evaluation results, plots, CSVs
├── logs/                 # Training logs (WandB / TensorBoard)
├── README.md             # You're here
├── pyproject.toml        # Poetry-managed dependencies
└── requirements.txt      # Exported for non-Poetry compatibility
💻 3. Quick Installation (with Poetry)
bash
Copier
Modifier
# Create environment
poetry install

# Activate environment
poetry shell
If needed:

bash
Copier
Modifier
conda create -n xaiguiformer python=3.10
conda activate xaiguiformer
pip install -r requirements.txt
⚙️ 4. EEG Preprocessing
Construct EEG connectomes per frequency band from raw TUAB or TDBRAIN datasets:

bash
Copier
Modifier
python data/preprocessing_scripts/eeg_preprocessing.py
🏋️ 5. Train the Model
Example: train on TUAB with full XAIguiFormer (dRoFE + XAI-guidance):

bash
Copier
Modifier
python scripts/train.py --config configs/tuab_config.yaml
📈 6. Evaluate a Trained Model
Example: evaluate a saved model on TUAB test set:

bash
Copier
Modifier
python scripts/evaluate.py --checkpoint checkpoints/tuab_best_model.pth
🧪 7. Run Ablation Studies
Example: disable dRoFE or XAI-guidance:

bash
Copier
Modifier
python scripts/ablation_study.py
🧠 8. Visualize Attention & Frequency Importance
Generate plots for:

Attention concentration (XAI vs vanilla)

Frequency band importance

bash
Copier
Modifier
python scripts/plot_attention_maps.py
📊 9. Reproducibility Outputs
All metrics (BAC, AUROC, AUC-PR) are saved in results/.

All model weights are stored in checkpoints/.

All figures are stored in results/figures/.

📌 Notes
All training is performed over 5 different random seeds to ensure statistical robustness.

All experimental configurations are versioned in configs/.

Baseline comparisons follow those reported in the original paper (FFCL, BIOT, S3T, etc.).

✅ You're Ready to Reproduce!
Feel free to fork this repository, run your own experiments, or use this codebase as a foundation for future EEG + XAI research.