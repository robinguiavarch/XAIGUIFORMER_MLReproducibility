#!/usr/bin/env python3
"""
Script to create a balanced dataset of 88 TDBRAIN patients:
- MDD: 22 patients
- ADHD: 22 patients
- SMC: 22 patients
- HEALTHY: 22 patients
- Stratified splits: Train (48), Val (20), Test (20)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_tdbrain_90_dataset():
    """
    Creates a dataset of 88 patients with balanced diagnostic distribution.

    Returns:
        pd.DataFrame or None: The final dataset or None if creation failed.
    """
    input_file = "TDBRAIN_participants_V2.csv"
    output_file = "TDBRAIN_90_patients.csv"

    target_distribution = {
        'MDD': 22,
        'ADHD': 22,
        'SMC': 22,
        'HEALTHY': 22
    }

    total_patients = sum(target_distribution.values())

    print("Creating TDBRAIN dataset with 88 patients")
    print(f"Working directory: {os.getcwd()}")
    print(f"Loading input file: {input_file}")

    try:
        df = pd.read_csv(input_file)
        print(f"File loaded: {len(df)} rows")
    except FileNotFoundError:
        print(f"Error: file not found: {input_file}")
        return None
    except Exception as e:
        print(f"Error during file loading: {e}")
        return None

    print("Cleaning data...")

    df_clean = df.dropna(subset=['indication']).copy()
    excluded_values = ['REPLICATION', 'UNKNOWN']
    df_clean = df_clean[~df_clean['indication'].isin(excluded_values)].copy()

    print(f"Cleaned dataset: {len(df_clean)} patients")

    diagnosis_counts = df_clean['indication'].value_counts()
    print("Checking availability of each diagnosis:")

    all_available = True
    for diagnosis, target_count in target_distribution.items():
        available = diagnosis_counts.get(diagnosis, 0)
        status = "OK" if available >= target_count else "Missing"
        print(f"{status} - {diagnosis}: {target_count} required, {available} available")
        if available < target_count:
            all_available = False

    if not all_available:
        print("Not enough patients for some diagnoses.")
        return None

    print("Selecting patients for each diagnosis...")
    selected_patients = []

    for diagnosis, target_count in target_distribution.items():
        diagnosis_patients = df_clean[df_clean['indication'] == diagnosis].copy()

        if len(diagnosis_patients) > target_count:
            try:
                if 'gender' in diagnosis_patients.columns:
                    selected = []
                    for gender in [0, 1]:
                        gender_patients = diagnosis_patients[diagnosis_patients['gender'] == gender]
                        if len(gender_patients) > 0:
                            n_to_select = min(len(gender_patients), target_count // 2)
                            selected.append(gender_patients.sample(n=n_to_select, random_state=42))

                    selected_df = pd.concat(selected, ignore_index=True)

                    if len(selected_df) < target_count:
                        remaining_needed = target_count - len(selected_df)
                        remaining_patients = diagnosis_patients[~diagnosis_patients.index.isin(selected_df.index)]
                        additional = remaining_patients.sample(n=remaining_needed, random_state=42)
                        selected_df = pd.concat([selected_df, additional], ignore_index=True)

                    selected_df = selected_df.sample(n=target_count, random_state=42)
                else:
                    selected_df = diagnosis_patients.sample(n=target_count, random_state=42)

            except:
                selected_df = diagnosis_patients.sample(n=target_count, random_state=42)
        else:
            selected_df = diagnosis_patients.copy()

        selected_patients.append(selected_df)
        print(f"{diagnosis}: {len(selected_df)} patients selected")

    final_dataset = pd.concat(selected_patients, ignore_index=True)

    print("Final dataset statistics:")
    print(f"Total patients: {len(final_dataset)}")
    final_distribution = final_dataset['indication'].value_counts()
    for diagnosis, count in final_distribution.items():
        print(f"{diagnosis}: {count} patients")

    print("Creating stratified splits...")

    X = final_dataset.drop('indication', axis=1)
    y = final_dataset['indication']

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            train_size=48,
            test_size=40,
            stratify=y,
            random_state=42
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=20,
            test_size=20,
            stratify=y_temp,
            random_state=42
        )

        final_dataset['split'] = 'unknown'
        final_dataset.loc[X_train.index, 'split'] = 'train'
        final_dataset.loc[X_val.index, 'split'] = 'val'
        final_dataset.loc[X_test.index, 'split'] = 'test'

        print("Splits successfully created:")
        split_counts = final_dataset['split'].value_counts()
        for split, count in split_counts.items():
            print(f"{split}: {count} patients")

        print("Diagnosis distribution per split:")
        split_diagnosis = pd.crosstab(final_dataset['split'], final_dataset['indication'])
        print(split_diagnosis)

    except ValueError as e:
        print(f"Stratification error: {e}")
        print("Fallback: random split without stratification")
        final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        final_dataset['split'] = (['train'] * 48 + ['val'] * 20 + ['test'] * 20)[:len(final_dataset)]

    print("Saving final dataset...")
    final_dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to: {output_file}")

    if 'age' in final_dataset.columns and 'gender' in final_dataset.columns:
        print("Demographic statistics:")
        print(f"Average age: {final_dataset['age'].mean():.1f} ± {final_dataset['age'].std():.1f}")
        print(f"Age range: {final_dataset['age'].min():.0f} - {final_dataset['age'].max():.0f}")
        gender_counts = final_dataset['gender'].value_counts()
        for gender, count in gender_counts.items():
            label = "Female" if gender == 0 else "Male"
            percentage = (count / len(final_dataset)) * 100
            print(f"{label}: {count} patients ({percentage:.1f}%)")

    return final_dataset

def main():
    """
    Main entry point for creating the TDBRAIN dataset.
    """
    print("Starting TDBRAIN 88-patient dataset creation")

    dataset = create_tdbrain_90_dataset()

    if dataset is not None:
        print("Dataset successfully created.")
        print("Ready for preprocessing_timeseries.py")

        print("Preview of the dataset:")
        available_cols = ['participants_ID', 'indication', 'age', 'gender', 'split']
        existing_cols = [col for col in available_cols if col in dataset.columns]
        if existing_cols:
            print(dataset[existing_cols].head(10))
        else:
            print("Available columns:", list(dataset.columns)[:10])
    else:
        print("Dataset creation failed.")

if __name__ == "__main__":
    main()
