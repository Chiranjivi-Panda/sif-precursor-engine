#!/usr/bin/env python
"""
Verify that all three raw data files are present and readable.

Run this ONCE after placing the renamed files into data/raw/.
Prints each file's path and shape (or open-confirmation for .xlsx).
Does NOT modify anything -- read-only check.

Usage:
    python src/verify_raw_files.py
"""

from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "safety_data_main.csv": {
        "role": "PRIMARY -- training data with Description column",
        "type": "csv",
    },
    "safety_data_no_description_IGNORE.csv": {
        "role": "REFERENCE ONLY -- no Description column, never load for training",
        "type": "csv",
    },
    "osha_template_reference_ONLY.xlsx": {
        "role": "REFERENCE ONLY -- blank OSHA-301 form template, zero data rows",
        "type": "xlsx",
    },
}


def main() -> None:
    print("=" * 70)
    print("  Raw Data File Verification")
    print("=" * 70)

    all_ok = True

    for filename, info in FILES.items():
        path = RAW_DIR / filename
        print(f"\n  [FILE] {filename}")
        print(f"     Role : {info['role']}")
        print(f"     Path : {path}")

        if not path.exists():
            print("     [MISSING] NOT FOUND -- please place this file in data/raw/")
            all_ok = False
            continue

        try:
            if info["type"] == "csv":
                full_df = pd.read_csv(path)
                rows, cols = full_df.shape
                print(f"     [OK] Found -- {rows:,} rows x {cols} columns")
                print(f"     Columns: {list(full_df.columns)}")
                if "Description" in full_df.columns:
                    print("     [INFO] Has 'Description' column")
                else:
                    print("     [WARN] No 'Description' column")
            elif info["type"] == "xlsx":
                xl = pd.ExcelFile(path)
                sheet_names = xl.sheet_names
                print(f"     [OK] Opens successfully -- sheets: {sheet_names}")
                # Try to read first sheet to confirm it's a blank template
                df_xl = xl.parse(sheet_names[0])
                print(f"     Shape: {df_xl.shape[0]} rows x {df_xl.shape[1]} columns")
        except Exception as e:
            print(f"     [ERROR] Error reading file: {e}")
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("  [OK] All files verified successfully.")
    else:
        print("  [WARN] Some files are missing or unreadable. See above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
