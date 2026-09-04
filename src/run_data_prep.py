#!/usr/bin/env python
"""
Quick-run script for the data preparation pipeline.

Usage:
    python src/run_data_prep.py

Loads the raw IHMStefanini CSV, runs the full cleaning + labelling
pipeline, and prints the summary to stdout.
"""

from config import RAW_CSV_PATH
from data_prep import load_and_prepare_data


def main() -> None:
    print("=" * 65)
    print("  SIF Precursor Engine -- Data Preparation")
    print("=" * 65)
    df = load_and_prepare_data(RAW_CSV_PATH)
    print(f"\n[OK] Done.  DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
