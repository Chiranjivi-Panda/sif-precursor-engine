"""
src/patch_source_column.py

One-off patch: adds a 'source' column to the existing labeled_data.csv.

WHY THIS EXISTS
---------------
data_prep.py now writes a 'source' column, but it cannot be re-run here
because data/raw/safety_data_main.csv is gitignored and not present locally.
Since labeled_data.csv is already correct in every other respect, we patch
the single missing column rather than regenerate the whole file.

Safe to run repeatedly -- it is idempotent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_CSV_PATH

df = pd.read_csv(PROCESSED_CSV_PATH)
print(f"Loaded {len(df)} rows from {PROCESSED_CSV_PATH}")

if "source" in df.columns:
    print(f"'source' column already present: {df['source'].unique().tolist()}")
else:
    df["source"] = "IHM_Stefanini"
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    print("Added 'source' = 'IHM_Stefanini' to all rows.")
    print(f"Saved -> {PROCESSED_CSV_PATH}")

print(f"\nColumns now: {list(df.columns)}")