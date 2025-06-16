#!/usr/bin/env python3
"""
Script pour créer un dataset de 90 patients TDBRAIN avec répartition équilibrée
- MDD: 22 patients
- ADHD: 22 patients  
- SMC: 22 patients
- HEALTHY: 22 patients
- Splits stratifiés: Train(50) / Val(20) / Test(20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_tdbrain_90_dataset():
    """Crée un dataset de 90 patients avec répartition équilibrée"""
    
    # Configuration
    input_file = "data/TDBRAIN_participants_V2.csv"
    output_file = "data/TDBRAIN_90_patients.csv"
    
    # Répartition cible (88 patients = 22×4)
    target_distribution = {
        'MDD': 22,
        'ADHD': 22, 
        'SMC': 22,
        'HEALTHY': 22
    }
    total_patients = sum(target_distribution.values())  # 88 patients
    
    print("=== Création du dataset TDBRAIN 88 patients ===")
    print(f"📖 Lecture du fichier: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Fichier chargé: {len(df)} lignes")
        
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {input_file}")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None
    
    # Nettoyer les données
    print(f"\n🧹 Nettoyage des données...")
    
    # Supprimer les NaN et valeurs indésirables
    df_clean = df.dropna(subset=['indication']).copy()
    excluded_values = ['REPLICATION', 'UNKNOWN']
    df_clean = df_clean[~df_clean['indication'].isin(excluded_values)].copy()
    
    print(f"  - Dataset nettoyé: {len(df_clean)} patients")
    
    # Vérifier la disponibilité des diagnostics
    diagnosis_counts = df_clean['indication'].value_counts()
    print(f"\n📊 Vérification de la disponibilité:")
    
    all_available = True
    for diagnosis, target_count in target_distribution.items():
        available = diagnosis_counts.get(diagnosis, 0)
        status = "✅" if available >= target_count else "❌"
        print(f"  {status} {diagnosis}: {target_count} demandés, {available} disponibles")
        
        if available < target_count:
            all_available = False
    
    if not all_available:
        print(f"\n❌ Pas assez de patients pour certains diagnostics!")
        return None
    
    # Sélectionner les patients pour chaque diagnostic
    print(f"\n🎯 Sélection des patients...")
    selected_patients = []
    
    for diagnosis, target_count in target_distribution.items():
        # Filtrer les patients avec ce diagnostic
        diagnosis_patients = df_clean[df_clean['indication'] == diagnosis].copy()
        
        # Sélection aléatoire stratifiée par âge et genre si possible
        if len(diagnosis_patients) > target_count:
            # Essayer une sélection stratifiée par genre
            try:
                if 'gender' in diagnosis_patients.columns:
                    # Stratifier par genre si possible
                    selected = []
                    for gender in [0, 1]:  # 0=Female, 1=Male
                        gender_patients = diagnosis_patients[diagnosis_patients['gender'] == gender]
                        if len(gender_patients) > 0:
                            n_to_select = min(len(gender_patients), target_count // 2)
                            selected.append(gender_patients.sample(n=n_to_select, random_state=42))
                    
                    selected_df = pd.concat(selected, ignore_index=True)
                    
                    # Compléter si nécessaire
                    if len(selected_df) < target_count:
                        remaining_needed = target_count - len(selected_df)
                        remaining_patients = diagnosis_patients[~diagnosis_patients.index.isin(selected_df.index)]
                        additional = remaining_patients.sample(n=remaining_needed, random_state=42)
                        selected_df = pd.concat([selected_df, additional], ignore_index=True)
                    
                    # Prendre exactement le nombre voulu
                    selected_df = selected_df.sample(n=target_count, random_state=42)
                    
                else:
                    # Sélection aléatoire simple
                    selected_df = diagnosis_patients.sample(n=target_count, random_state=42)
                    
            except:
                # Fallback: sélection aléatoire simple
                selected_df = diagnosis_patients.sample(n=target_count, random_state=42)
        else:
            # Prendre tous les patients disponibles
            selected_df = diagnosis_patients.copy()
        
        selected_patients.append(selected_df)
        print(f"  - {diagnosis}: {len(selected_df)} patients sélectionnés")
    
    # Combiner tous les patients sélectionnés
    final_dataset = pd.concat(selected_patients, ignore_index=True)
    
    print(f"\n📊 Dataset final:")
    print(f"  - Total: {len(final_dataset)} patients")
    
    # Vérifier la répartition finale
    final_distribution = final_dataset['indication'].value_counts()
    for diagnosis, count in final_distribution.items():
        print(f"  - {diagnosis}: {count} patients")
    
    # Créer les splits stratifiés
    print(f"\n🎲 Création des splits stratifiés...")
    
    # Préparer les labels pour la stratification
    X = final_dataset.drop('indication', axis=1)
    y = final_dataset['indication']
    
    try:
        # Premier split: Train (48) vs Temp (40)  
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            train_size=48,
            test_size=40,
            stratify=y,
            random_state=42
        )
        
        # Deuxième split: Val (20) vs Test (20)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=20,
            test_size=20,
            stratify=y_temp,
            random_state=42
        )
        
        # Ajouter les colonnes de split
        final_dataset['split'] = 'unknown'
        final_dataset.loc[X_train.index, 'split'] = 'train'
        final_dataset.loc[X_val.index, 'split'] = 'val'
        final_dataset.loc[X_test.index, 'split'] = 'test'
        
        print(f"  ✅ Splits créés avec succès:")
        split_counts = final_dataset['split'].value_counts()
        for split, count in split_counts.items():
            print(f"    - {split}: {count} patients")
        
        # Vérifier la stratification par split
        print(f"\n📋 Répartition par diagnostic et split:")
        split_diagnosis = pd.crosstab(final_dataset['split'], final_dataset['indication'])
        print(split_diagnosis)
        
    except ValueError as e:
        print(f"❌ Erreur lors de la stratification: {e}")
        print(f"💡 Fallback: split aléatoire sans stratification")
        
        # Fallback: split aléatoire
        final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        total_patients = len(final_dataset)  # 88
        final_dataset['split'] = (['train'] * 48 + ['val'] * 20 + ['test'] * 20)[:total_patients]
    
    # Sauvegarder le dataset
    print(f"\n💾 Sauvegarde du dataset...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Sauvegarder
    final_dataset.to_csv(output_file, index=False)
    print(f"✅ Dataset sauvegardé: {output_file}")
    
    # Statistiques démographiques
    if 'age' in final_dataset.columns and 'gender' in final_dataset.columns:
        print(f"\n👥 Statistiques démographiques:")
        print(f"  - Âge moyen: {final_dataset['age'].mean():.1f} ± {final_dataset['age'].std():.1f} ans")
        print(f"  - Âge min-max: {final_dataset['age'].min():.0f} - {final_dataset['age'].max():.0f} ans")
        
        gender_counts = final_dataset['gender'].value_counts()
        for gender, count in gender_counts.items():
            gender_label = "Femme" if gender == 0 else "Homme"
            percentage = (count / len(final_dataset)) * 100
            print(f"  - {gender_label}: {count} patients ({percentage:.1f}%)")
    
    return final_dataset

def main():
    """Fonction principale"""
    print("🚀 Démarrage de la création du dataset TDBRAIN 88 patients")
    
    dataset = create_tdbrain_90_dataset()
    
    if dataset is not None:
        print(f"\n🎉 Dataset créé avec succès!")
        print(f"Prêt pour preprocessing_timeseries.py")
        
        # Afficher un aperçu
        print(f"\n👀 Aperçu du dataset:")
        # Utiliser les vraies colonnes du CSV
        available_cols = ['participants_ID', 'indication', 'age', 'gender', 'split']
        # Vérifier quelles colonnes existent réellement
        existing_cols = [col for col in available_cols if col in dataset.columns]
        if len(existing_cols) > 0:
            print(dataset[existing_cols].head(10))
        else:
            print("Colonnes disponibles:", list(dataset.columns)[:10])
        
    else:
        print(f"\n💥 Échec de la création du dataset")

if __name__ == "__main__":
    main()