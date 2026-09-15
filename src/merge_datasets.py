"""
src/merge_datasets.py

Builds the final training corpus from four sources with an explicit,
controllable SIF ratio.

  REAL POSITIVES : IHM Stefanini (IV-VI)  +  OSHA SIR  +  OSHA Abstracts
  REAL NEGATIVES : IHM Stefanini (I-III)
  SYNTH NEGATIVES: templated minor incidents (TRAIN-ONLY, tagged in `source`)

Two leakage guards are enforced here:
  1. Every row from every source gets IDENTICAL placeholder metadata, so the
     model cannot infer the label from Industry Sector / Employee type.
     (This is the bug that made "Metals + Third Party" flip a 99% SIF report
      down to 31%.)
  2. Synthetic rows carry source='SYNTHETIC_NEG' so the trainer can hold them
     out of every test fold.
"""

from pathlib import Path
import pandas as pd

P = Path("data/processed")
ORIGINAL = P / "labeled_data.csv"
OSHA_SIR = P / "osha_cleaned.csv"
OSHA_ABS = P / "osha_abstracts_positives.csv"
SYNTH_NEG = P / "synthetic_negatives.csv"
OUTPUT = P / "labeled_data_expanded.csv"

# ---- TUNE THESE ----
N_OSHA_SIR = 450        # severe-injury reports (amputation/hospitalisation)
N_OSHA_ABS = 450        # accident-investigation abstracts (many fatalities)
USE_SYNTHETIC = True    # set False to train on real data only
N_SYNTHETIC = 900
SEED = 42

STD_COLS = ["Description", "is_sif", "source",
            "Industry Sector", "Employee or Third Party", "iogp_rule"]


def standardise(df, source_name=None):
    """Force every source onto one schema with identical neutral metadata."""
    out = pd.DataFrame()
    out["Description"] = df["Description"].astype(str).str.strip()
    out["is_sif"] = df["is_sif"].astype(int)
    out["source"] = df["source"] if "source" in df.columns else source_name
    # identical for all rows -> metadata carries zero label information
    out["Industry Sector"] = "Others"
    out["Employee or Third Party"] = "Employee"
    out["iogp_rule"] = df["iogp_rule"] if "iogp_rule" in df.columns else "Uncategorized"
    return out[STD_COLS]


def load(path, n=None, label=None, src=None):
    if not path.exists():
        print(f"   [skip] {path.name} not found")
        return pd.DataFrame(columns=STD_COLS)
    df = pd.read_csv(path)
    if label is not None:
        df["is_sif"] = label
    if src is not None:
        df["source"] = src
    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=SEED)
    return standardise(df, src)


def main():
    print("Loading sources...\n")

    base = pd.read_csv(ORIGINAL)
    base["source"] = "IHM_Stefanini"
    base = standardise(base)
    base_pos = base[base.is_sif == 1]
    base_neg = base[base.is_sif == 0]
    print(f"   IHM Stefanini      : {len(base_pos)} pos / {len(base_neg)} neg")

    sir = load(OSHA_SIR, n=N_OSHA_SIR, label=1, src="OSHA_SIR")
    print(f"   OSHA SIR           : {len(sir)} pos")

    abs_ = load(OSHA_ABS, n=N_OSHA_ABS, label=1, src="OSHA_ABSTRACTS")
    print(f"   OSHA Abstracts     : {len(abs_)} pos")

    parts = [base_pos, base_neg, sir, abs_]

    if USE_SYNTHETIC:
        syn = load(SYNTH_NEG, n=N_SYNTHETIC, label=0, src="SYNTHETIC_NEG")
        print(f"   Synthetic negatives: {len(syn)} neg  (TRAIN-ONLY)")
        parts.append(syn)

    combined = pd.concat(parts, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["Description"])
    combined = combined[combined["Description"].str.len() >= 20]
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"\n   Removed {before - len(combined)} duplicate/short rows")

    n = len(combined)
    pos = int(combined.is_sif.sum())
    neg = n - pos
    real = combined[combined.source != "SYNTHETIC_NEG"]
    r_pos = int(real.is_sif.sum())
    r_neg = len(real) - r_pos

    print("\n" + "=" * 58)
    print("               FINAL TRAINING CORPUS")
    print("=" * 58)
    print(f" Total rows        : {n:,}")
    print(f"   SIF (1)         : {pos:,}  ({pos/n*100:.1f}%)")
    print(f"   Non-SIF (0)     : {neg:,}  ({neg/n*100:.1f}%)")
    print(f"\n REAL rows only (what metrics will be reported on):")
    print(f"   Total           : {len(real):,}")
    print(f"   SIF (1)         : {r_pos:,}  ({r_pos/len(real)*100:.1f}%)")
    print(f"   Non-SIF (0)     : {r_neg:,}  ({r_neg/len(real)*100:.1f}%)")
    print("\n By source:")
    for s, c in combined.source.value_counts().items():
        print(f"   {s:<18}: {c:,}")
    print("=" * 58)

    combined.to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()