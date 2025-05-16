# XAIguiFormer Reproducibility Challenge

## 1. Introduction

Reproduction du papier :  
**"XAIGUIFORMER: Explainable Artificial Intelligence Guided Transformer for Brain Disorder Identification"** (ICLR 2025).

Notre objectif est de reproduire intégralement les méthodes et les résultats du papier, dans le cadre du ML Reproducibility Challenge.

---

## 2. Organisation du dépôt

configs/ # Fichiers de configuration YAML pour chaque expérience
src/ # Code source : modèles, tokenizer, explainers, losses
scripts/ # Scripts pour entraînement, évaluation, ablations, visualisations
data/ # Scripts de prétraitement EEG et construction des connectomes
checkpoints/ # Modèles sauvegardés
results/ # Fichiers CSV des résultats reproductibles
logs/ # Logs d'entraînement (TensorBoard ou WandB)

---

## 3. Installation

```bash
conda create -n xaiguiformer python=3.10
conda activate xaiguiformer
pip install -r requirements.txt

4. Prétraitement des données EEG

Construction des connectomes EEG multi-bandes à partir des datasets bruts **TUAB** et **TDBRAIN** :

```bash
python data/preprocessing_scripts/eeg_preprocessing.py
5. Entraîner un modèle
Exemple : Entraîner XAIguiFormer sur TUAB avec dRoFE + XAI-guidance :

bash
Copier
Modifier
python scripts/train.py --config configs/tuab_config.yaml
6. Évaluer un modèle
Exemple : Évaluer un modèle sauvegardé sur le jeu de test TUAB :

bash
Copier
Modifier
python scripts/evaluate.py --checkpoint checkpoints/tuab_best_model.pth
7. Reproduire les études d'ablation
Exemple : Reproduire une ablation (sans dRoFE ou sans XAI-guidance) :

bash
Copier
Modifier
python scripts/ablation_study.py
8. Visualiser l'attention et l'importance des bandes de fréquences
Générer les figures d'attention affinée et d'importance fréquentielle :

bash
Copier
Modifier
python scripts/plot_attention_maps.py
9. Résultats reproductibles
Toutes les métriques (BAC, AUROC, AUC-PR) sont sauvegardées dans le dossier results/.

Tous les modèles entraînés sont sauvegardés dans checkpoints/.

Les figures générées sont stockées dans results/figures/.

📌 Notes complémentaires
Tous les entraînements sont réalisés sur 5 seeds pour garantir la robustesse des résultats et permettre de calculer moyenne ± écart-type.

Tous les paramètres d’expériences sont versionnés dans le dossier configs/.

Les résultats expérimentaux sont systématiquement comparés aux méthodes de référence (baselines) mentionnées dans l'article original.