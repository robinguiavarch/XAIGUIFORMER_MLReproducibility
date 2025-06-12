#!/usr/bin/env python3
"""
Script pour diagnostiquer et corriger le problème Gender NaN
qui cause l'explosion du modèle dans dRoFE.
"""

import pandas as pd
import pickle
import os

def diagnose_participants_file():
    """Diagnostique le fichier participants.tsv"""
    print("=== DIAGNOSTIC PARTICIPANTS.TSV ===")
    
    tsv_path = "data/TDBRAIN/raw/participants.tsv"
    if not os.path.exists(tsv_path):
        print(f"❌ Fichier non trouvé: {tsv_path}")
        return None
    
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Total lignes: {len(df)}")
    print(f"Colonnes: {df.columns.tolist()}")
    
    # Analyser chaque colonne critique
    for col in ['participant_id', 'indication', 'age', 'gender']:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  - NaN count: {df[col].isna().sum()}")
            print(f"  - Unique values: {df[col].unique()}")
        else:
            print(f"❌ Colonne manquante: {col}")
    
    print(f"\nPremières lignes:")
    print(df.head())
    
    return df

def diagnose_xai_graphs():
    """Diagnostique le fichier xai_graphs.pkl"""
    print("\n=== DIAGNOSTIC XAI_GRAPHS.PKL ===")
    
    pkl_path = "data/TDBRAIN/tokens/xai_graphs.pkl"
    if not os.path.exists(pkl_path):
        print(f"❌ Fichier non trouvé: {pkl_path}")
        return None
    
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    
    print(f"Nombre de graphes: {len(graphs)}")
    
    # Analyser les valeurs NaN
    nan_ages = sum(1 for g in graphs if g.age.isnan().any())
    nan_genders = sum(1 for g in graphs if g.gender.isnan().any())
    
    print(f"Graphes avec age NaN: {nan_ages}")
    print(f"Graphes avec gender NaN: {nan_genders}")
    
    # Afficher quelques exemples
    for i, g in enumerate(graphs[:3]):
        print(f"\nGraphe {i}:")
        print(f"  - subject_id: {g.subject_id}")
        print(f"  - age: {g.age}")
        print(f"  - gender: {g.gender}")
        print(f"  - y: {g.y}")
    
    return graphs

def fix_connectome_builder():
    """Suggère la correction pour connectome_builder.py"""
    print("\n=== CORRECTION SUGGÉRÉE ===")
    print("""
Dans src/preprocessing/connectome_builder.py, remplacez la fonction load_labels():

def load_labels(tsv_path="data/TDBRAIN/raw/participants.tsv"):
    df = pd.read_csv(tsv_path, sep="\\t")
    
    # ✅ CORRECTION : Nettoyer d'abord les NaN
    print(f"Avant nettoyage - Total: {len(df)}")
    print(f"NaN dans indication: {df['indication'].isna().sum()}")
    print(f"NaN dans age: {df['age'].isna().sum()}")
    print(f"NaN dans gender: {df['gender'].isna().sum()}")
    
    # Supprimer les lignes avec des NaN dans les colonnes critiques
    df = df.dropna(subset=["indication", "age", "gender"])
    print(f"Après nettoyage - Total: {len(df)}")
    
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["indication"])

    # ✅ SÉCURITÉ : Vérifier les valeurs gender avant mapping
    print(f"Valeurs gender uniques: {df['gender'].unique()}")
    
    # Normaliser le genre (M/F en 0/1)
    gender_mapping = {"M": 1.0, "F": 0.0}
    df["gender"] = df["gender"].map(gender_mapping)
    
    # ✅ VÉRIFICATION : Pas de NaN après mapping
    if df["gender"].isna().sum() > 0:
        print(f"❌ ATTENTION: {df['gender'].isna().sum()} NaN après gender mapping!")
        print(f"Valeurs non mappées: {df[df['gender'].isna()]['gender'].unique()}")
        # Option: fillna(0.5) ou supprimer ces lignes
        df = df.dropna(subset=["gender"])
    
    # Créer mapping pour chaque attribut
    label_map = dict(zip(df["participant_id"], df["label"]))
    age_map = dict(zip(df["participant_id"], df["age"]))
    gender_map = dict(zip(df["participant_id"], df["gender"]))
    
    print(f"Mappings créés pour {len(label_map)} participants")
    return label_map, age_map, gender_map, le
""")

if __name__ == "__main__":
    # 1. Diagnostiquer le fichier source
    df = diagnose_participants_file()
    
    # 2. Diagnostiquer les graphes générés
    graphs = diagnose_xai_graphs()
    
    # 3. Suggérer les corrections
    fix_connectome_builder()
    
    print("\n" + "="*50)
    print("ACTIONS À FAIRE:")
    print("1. Corrigez connectome_builder.py avec le code suggéré")
    print("2. Relancez: python src/preprocessing/connectome_builder.py")
    print("3. Relancez: python src/preprocessing/aggregate_tokens.py")
    print("4. Relancez debug_dimensions.py pour vérifier")
    print("="*50)