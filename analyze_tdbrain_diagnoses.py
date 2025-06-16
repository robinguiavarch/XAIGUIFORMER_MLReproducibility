#!/usr/bin/env python3
"""
Script pour analyser les diagnostics dans TDBRAIN_participants_V2.csv
- Supprime les NaN, "REPLICATION" et "UNKNOWN"
- Identifie les 4 diagnostics les plus fréquents
- Compte le nombre de patients pour chaque diagnostic
"""

import pandas as pd
import numpy as np

def analyze_tdbrain_diagnoses():
    """Analyse les diagnostics du fichier TDBRAIN"""
    
    # Charger le fichier CSV
    csv_file = "data/TDBRAIN_participants_V2.csv"
    
    print("=== Analyse des diagnostics TDBRAIN ===")
    print(f"📖 Lecture du fichier: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Fichier chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {csv_file}")
        return
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # Vérifier que la colonne 'indication' existe
    if 'indication' not in df.columns:
        print("❌ Colonne 'indication' non trouvée!")
        print(f"Colonnes disponibles: {list(df.columns)}")
        return
    
    print(f"\n📊 Données originales:")
    print(f"  - Total des lignes: {len(df)}")
    print(f"  - Valeurs dans 'indication': {df['indication'].notna().sum()}")
    print(f"  - Valeurs NaN: {df['indication'].isna().sum()}")
    
    # Afficher les valeurs uniques avant nettoyage
    print(f"\n🔍 Valeurs uniques dans 'indication' (avant nettoyage):")
    unique_values = df['indication'].value_counts(dropna=False)
    for value, count in unique_values.items():
        print(f"  - '{value}': {count} patients")
    
    # Nettoyer les données
    print(f"\n🧹 Nettoyage des données...")
    
    # Supprimer les NaN
    df_clean = df.dropna(subset=['indication']).copy()
    print(f"  - Après suppression des NaN: {len(df_clean)} lignes")
    
    # Supprimer "REPLICATION" et "UNKNOWN"
    excluded_values = ['REPLICATION', 'UNKNOWN']
    df_clean = df_clean[~df_clean['indication'].isin(excluded_values)].copy()
    print(f"  - Après suppression de {excluded_values}: {len(df_clean)} lignes")
    
    # Compter les diagnostics après nettoyage
    print(f"\n📈 Diagnostics après nettoyage:")
    diagnosis_counts = df_clean['indication'].value_counts()
    
    for i, (diagnosis, count) in enumerate(diagnosis_counts.items(), 1):
        print(f"  {i}. '{diagnosis}': {count} patients")
    
    # Identifier les 4 diagnostics les plus fréquents
    top_4_diagnoses = diagnosis_counts.head(4)
    
    print(f"\n🏆 TOP 4 diagnostics les plus fréquents:")
    total_patients_top4 = 0
    
    for i, (diagnosis, count) in enumerate(top_4_diagnoses.items(), 1):
        percentage = (count / len(df_clean)) * 100
        total_patients_top4 += count
        print(f"  {i}. '{diagnosis}': {count} patients ({percentage:.1f}%)")
    
    print(f"\n📊 Résumé:")
    print(f"  - Total patients après nettoyage: {len(df_clean)}")
    print(f"  - Patients dans TOP 4 diagnostics: {total_patients_top4}")
    print(f"  - Autres diagnostics: {len(df_clean) - total_patients_top4} patients")
    
    # Recommandations pour 90 patients
    print(f"\n💡 Recommandations pour 90 patients:")
    
    if total_patients_top4 >= 90:
        print(f"  ✅ Les 4 diagnostics principaux ({total_patients_top4} patients) suffisent pour 90 patients")
        
        # Calculer une répartition équilibrée
        print(f"\n📋 Répartition suggérée pour 90 patients:")
        for i, (diagnosis, count) in enumerate(top_4_diagnoses.items(), 1):
            suggested = min(count, 90 // 4)  # Répartition équilibrée
            print(f"  - {diagnosis}: {suggested} patients (disponible: {count})")
            
    else:
        print(f"  ⚠️  Les 4 diagnostics principaux ({total_patients_top4} patients) ne suffisent pas pour 90 patients")
        print(f"  💡 Considérer plus de diagnostics ou réduire l'objectif")
    
    # Sauvegarder le dataset nettoyé (optionnel)
    output_file = "data/TDBRAIN_participants_V2_clean.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"\n💾 Dataset nettoyé sauvegardé: {output_file}")
    
    return df_clean, top_4_diagnoses

def main():
    """Fonction principale"""
    print("🔬 Démarrage de l'analyse des diagnostics TDBRAIN")
    
    df_clean, top_4_diagnoses = analyze_tdbrain_diagnoses()
    
    if df_clean is not None:
        print(f"\n🎉 Analyse terminée avec succès!")
        print(f"Dataset nettoyé prêt pour preprocessing_timeseries.py")
    else:
        print(f"\n💥 Échec de l'analyse")

if __name__ == "__main__":
    main()