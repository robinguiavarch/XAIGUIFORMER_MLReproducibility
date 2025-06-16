#!/usr/bin/env python3
"""
Script pour convertir TDBRAIN_participants_V2.xlsx en CSV
"""

import pandas as pd
import os
import sys
from pathlib import Path

def convert_xlsx_to_csv():
    """Convertit le fichier XLSX en CSV"""
    
    # Chemins des fichiers
    input_file = "data/TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.xlsx"
    output_file = "data/TDBRAIN_participants_V2.csv"
    
    print("=== Conversion XLSX vers CSV ===")
    print(f"Fichier source: {input_file}")
    print(f"Fichier destination: {output_file}")
    
    # Vérifier si le fichier source existe
    if not os.path.exists(input_file):
        print(f"❌ ERREUR: Le fichier {input_file} n'existe pas")
        return False
    
    try:
        # Lire le fichier XLSX
        print("📖 Lecture du fichier XLSX...")
        df = pd.read_excel(input_file)
        
        # Créer le répertoire de destination si nécessaire
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder en CSV
        print("💾 Sauvegarde en CSV...")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Conversion réussie: {output_file}")
        
        # Afficher les informations sur le fichier généré
        print("\n=== Informations sur le fichier CSV généré ===")
        file_size = os.path.getsize(output_file) / 1024  # en KB
        print(f"Taille: {file_size:.1f} KB")
        print(f"Nombre de lignes: {len(df)}")
        print(f"Nombre de colonnes: {len(df.columns)}")
        
        print("\nColonnes disponibles:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        print("\nAperçu des premières lignes:")
        print(df.head())
        
        return True
        
    except ImportError as e:
        print("❌ Erreur: Dépendances manquantes")
        print("Installez les dépendances avec:")
        print("pip install pandas openpyxl")
        return False
        
    except Exception as e:
        print(f"❌ Erreur lors de la conversion: {e}")
        return False

def main():
    """Fonction principale"""
    print("🔄 Démarrage de la conversion XLSX → CSV")
    
    success = convert_xlsx_to_csv()
    
    if success:
        print("\n🎉 Conversion terminée avec succès!")
        print("Le fichier CSV est prêt pour preprocessing_timeseries.py")
    else:
        print("\n💥 Échec de la conversion")
        sys.exit(1)

if __name__ == "__main__":
    main()