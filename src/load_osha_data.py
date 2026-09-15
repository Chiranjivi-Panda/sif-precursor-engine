"""
src/load_osha_data.py
Loads, cleans, and standardizes the OSHA Severe Injury Report dataset
for the SIF-Sentinel precursor detection pipeline.
"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/osha_severe_injury_raw.csv")
OUTPUT_PATH = Path("data/processed/osha_cleaned.csv")
NARRATIVE_COL = "Final Narrative"


def main():
    print(f"1. Reading raw OSHA dataset from {RAW_PATH}...")
    # OSHA data often contains special characters; latin-1 prevents decode errors
    df = pd.read_csv(
        RAW_PATH,
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip",
    )
    print(f"   Loaded {len(df):,} raw rows.")

    print("2. Filtering and cleaning narratives...")
    # Drop rows without text
    df = df[df[NARRATIVE_COL].notna()].copy()

    # Clean whitespace and strip strings
    df[NARRATIVE_COL] = df[NARRATIVE_COL].astype(str).str.strip()

    # Drop narratives that are too short to be useful (e.g. fewer than 20 characters)
    df = df[df[NARRATIVE_COL].str.len() >= 20]

    # Build standardized dataframe matching our pipeline schema
    cleaned = pd.DataFrame()
    cleaned["Description"] = df[NARRATIVE_COL]
    cleaned["is_sif"] = 1  # All OSHA severe injury reports represent SIF outcomes
    cleaned["source"] = "OSHA"

    # Retain industry info for downstream analysis if present
    if "Primary NAICS" in df.columns:
        cleaned["industry_code"] = df["Primary NAICS"]
    elif "Industry" in df.columns:
        cleaned["industry_code"] = df["Industry"]
    else:
        cleaned["industry_code"] = "Unknown"

    # Fill any placeholder metadata to keep schema consistent
    cleaned["Industry Sector"] = "Others"
    cleaned["Employee or Third Party"] = "Employee"

    # Deduplicate exact duplicate narratives
    initial_count = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["Description"]).reset_index(drop=True)
    dropped_dupes = initial_count - len(cleaned)

    print(f"   Dropped {dropped_dupes:,} duplicate narratives.")
    print(f"   Cleaned narrative count: {len(cleaned):,} rows.")

    # Save to data/processed/
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"3. Successfully saved cleaned data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()