"""
Data preparation pipeline for the SIF Precursor Engine.

Loads the raw Kaggle/IHMStefanini industrial-safety CSV, cleans it,
applies the IOGP mapping, creates the binary SIF label, and saves the
result to data/processed/labeled_data.csv.

NOTE -- Genre, Accident Level, Potential Accident Level, and Critical Risk
        are **label sources only**. They must NOT be used as model features
        anywhere downstream.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from config import IOGP_MAP, SIF_LEVELS, PROCESSED_CSV_PATH


# Columns that must be non-null for every row.
REQUIRED_COLUMNS: list[str] = [
    "Description",
    "Potential Accident Level",
    "Critical Risk",
    "Local",
    "Industry Sector",
    "Employee or Third Party",
]


def load_and_prepare_data(raw_csv_path: str | Path) -> pd.DataFrame:
    """Load, clean, label, summarise, and persist the dataset.

    Parameters
    ----------
    raw_csv_path : str | Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        The cleaned and labelled dataframe.
    """
    raw_csv_path = Path(raw_csv_path)
    print(f"[data_prep] Loading raw CSV from: {raw_csv_path}")

    # -- 1. Load -------------------------------------------------------
    df = pd.read_csv(raw_csv_path)
    print(f"[data_prep] Raw shape: {df.shape}")

    # -- 2. Drop artefact index column ---------------------------------
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("[data_prep] Dropped 'Unnamed: 0' column.")

    # -- 3. Parse dates ------------------------------------------------
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
        print("[data_prep] Parsed 'Data' column to datetime.")

    # -- 4. Normalise description text ---------------------------------
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    df["Description"] = df["Description"].apply(_clean_text)
    print("[data_prep] Normalised whitespace in 'Description'.")

    # -- 5. Assert no missing values in required columns ---------------
    has_issue = False
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            print(f"  [WARN] Required column '{col}' is MISSING from the dataset!")
            has_issue = True
            continue
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"  [WARN] Column '{col}' has {n_null} missing value(s)!")
            has_issue = True
    if not has_issue:
        print("[data_prep] [OK] Zero missing values in all required columns.")

    # -- 6. Binary SIF label -------------------------------------------
    # NOTE -- 'Potential Accident Level' is a *label source* only.
    #         Do NOT use it as a model feature.
    df["is_sif"] = df["Potential Accident Level"].isin(SIF_LEVELS).astype(int)

    # -- 7. IOGP rule mapping ------------------------------------------
    # NOTE -- 'Critical Risk' is a *label source* only.
    #         Do NOT use it as a model feature.
    df["iogp_rule"] = df["Critical Risk"].map(
        lambda x: IOGP_MAP.get(x, "Uncategorized")
    )

    # -- 8. Provenance tag ---------------------------------------------
    # Every row in this file comes from the IHM Stefanini dataset. The column
    # exists so downstream code (predict.py nearest-neighbour display,
    # precompute_scores.py) can always report where a report originated.
    # It also keeps the schema stable if additional sources are ever
    # re-introduced after passing the src/diagnose_signal.py checks.
    df["source"] = "IHM_Stefanini"
    print("[data_prep] Tagged all rows with source = 'IHM_Stefanini'.")

    # -- 9. Summary ----------------------------------------------------
    _print_summary(df)

    # -- 10-11. Save and return ----------------------------------------
    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"\n[data_prep] Saved labelled data to: {PROCESSED_CSV_PATH}")

    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of labels and distribution."""
    n = len(df)
    print("\n" + "=" * 65)
    print(f"  DATASET SUMMARY  --  {n:,} rows")
    print("=" * 65)

    # -- is_sif distribution -------------------------------------------
    sif_counts = df["is_sif"].value_counts().sort_index()
    print("\n  is_sif distribution:")
    for val, cnt in sif_counts.items():
        pct = cnt / n * 100
        label = "SIF" if val == 1 else "non-SIF"
        print(f"    {val} ({label:>7s}): {cnt:>5,}  ({pct:5.1f}%)")

    # -- iogp_rule distribution ----------------------------------------
    rule_counts = df["iogp_rule"].value_counts()
    print("\n  iogp_rule distribution (before Model B merge):")
    for rule, cnt in rule_counts.items():
        pct = cnt / n * 100
        flag = "  << LOW COUNT" if (rule != "Uncategorized" and cnt < 10) else ""
        print(f"    {rule:<30s}: {cnt:>5,}  ({pct:5.1f}%){flag}")

    # Explicit low-count warning block
    low_cats = {
        rule: cnt
        for rule, cnt in rule_counts.items()
        if rule != "Uncategorized" and cnt < 10
    }
    if low_cats:
        print("\n  [WARN] The following IOGP categories have < 10 rows,")
        print("         which is too few to train/test reliably.")
        print("         train_model_b.py merges these -- see IOGP_MERGE in config.py:")
        for rule, cnt in low_cats.items():
            print(f"           * {rule}: {cnt}")

    print("=" * 65)