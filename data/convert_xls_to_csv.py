#!/usr/bin/env python3
"""
Script to convert TDBRAIN_participants_V2.xlsx to CSV
"""

import pandas as pd
import os
import sys
from pathlib import Path

def convert_xlsx_to_csv():
    """
    Convert the input XLSX file to a CSV file.

    Returns:
        bool: True if the conversion succeeds, False otherwise.
    """
    input_file = "TDBRAIN_participants_V2_data/TDBRAIN_participants_V2.xlsx"
    output_file = "TDBRAIN_participants_V2.csv"

    if not os.path.exists(input_file):
        print(f"Error: input file {input_file} does not exist.")
        return False

    try:
        df = pd.read_excel(input_file)

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_file, index=False)

        print(f"Conversion succeeded: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("Column names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")

        print("Preview of first rows:")
        print(df.head())

        return True

    except ImportError:
        print("Error: missing dependencies. Install them using 'pip install pandas openpyxl'")
        return False

    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def main():
    """
    Main entry point of the script.
    """
    print(f"Working directory: {os.getcwd()}")

    success = convert_xlsx_to_csv()

    if success:
        print("CSV file is ready.")
    else:
        print("Conversion failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
