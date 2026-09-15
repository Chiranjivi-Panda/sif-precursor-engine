"""
src/load_osha_abstracts.py

Loads the OSHA HSE Accident Abstracts dataset (2015-2017) and splits it into
SIF-positive and non-SIF rows using a conservative, documented severity rule.

IMPORTANT DESIGN NOTE:
  'Degree of Injury' == 'Nonfatal' is NOT sufficient to call something non-SIF.
  This dataset labels finger amputations as "Nonfatal". We therefore combine
  Degree of Injury + Nature of Injury + Event Type + fall height, and DROP any
  row we cannot confidently classify. Dropping ambiguous rows is deliberate:
  a wrong negative label is far more damaging than a smaller dataset.
"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/osha_abstracts_raw.csv")
POS_OUT = Path("data/processed/osha_abstracts_positives.csv")
NEG_OUT = Path("data/processed/osha_abstracts_negatives.csv")

TEXT_COL = "Abstract Text"
DEGREE_COL = "Degree of Injury"
NATURE_COL = "Nature of Injury"
EVENT_COL = "Event type"
FALLHT_COL = "fall_ht"

# --- Severity vocabularies (lowercase substring matching) ---

SEVERE_NATURE = [
    "amputation", "crushing", "fracture", "electric", "electrocution",
    "asphyxia", "suffocat", "drowning", "concussion", "internal",
    "poison", "burn", "chemical", "poisoning", "heat stroke",
    "hypothermia", "avulsion", "traumatic",
]

MINOR_NATURE = [
    "contusion", "bruise", "laceration", "cut", "abrasion", "scratch",
    "sprain", "strain", "foreign body", "dermatitis", "irritation",
    "inflammation", "blister", "heat exhaustion", "bite",
]

# Event types that indicate high-energy hazards. Rows matching these are
# NEVER allowed into the negative class, regardless of the actual injury.
HIGH_ENERGY_EVENTS = [
    "fall", "caught in", "caught between", "electrocution", "shock",
    "struck by", "struck against", "explosion", "fire", "cave-in",
    "engulf", "inhalation", "absorption", "ingestion", "crush",
]


def contains_any(text: str, vocab: list) -> bool:
    t = str(text).lower()
    return any(k in t for k in vocab)


def main():
    print(f"1. Reading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, encoding="latin-1",
                     low_memory=False, on_bad_lines="skip")
    print(f"   Loaded {len(df):,} rows.")

    # --- Basic text cleaning ---
    df = df[df[TEXT_COL].notna()].copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() >= 40]
    print(f"   {len(df):,} rows remain after dropping empty/short narratives.")

    # --- Show what's actually in the severity columns (diagnostic) ---
    print("\n2. Column distributions (top values):")
    for col in [DEGREE_COL, NATURE_COL, EVENT_COL]:
        if col in df.columns:
            print(f"\n   --- {col} ---")
            print(df[col].value_counts().head(12).to_string())

    # --- Apply the labeling rule ---
    print("\n3. Applying conservative severity rule ...")

    degree = df.get(DEGREE_COL, pd.Series([""] * len(df))).astype(str).str.lower()
    nature = df.get(NATURE_COL, pd.Series([""] * len(df))).astype(str)
    event = df.get(EVENT_COL, pd.Series([""] * len(df))).astype(str)

    if FALLHT_COL in df.columns:
        fall_ht = pd.to_numeric(df[FALLHT_COL], errors="coerce").fillna(0)
    else:
        fall_ht = pd.Series([0] * len(df), index=df.index)

    is_fatal = degree.str.contains("fatal") & ~degree.str.contains("nonfatal")
    severe_nature = nature.apply(lambda x: contains_any(x, SEVERE_NATURE))
    minor_nature = nature.apply(lambda x: contains_any(x, MINOR_NATURE))
    high_energy = event.apply(lambda x: contains_any(x, HIGH_ENERGY_EVENTS))

    df["_is_positive"] = is_fatal | severe_nature
    df["_is_negative"] = (
        (~df["_is_positive"])
        & minor_nature
        & (~high_energy)
        & (fall_ht <= 0)
    )

    positives = df[df["_is_positive"]].copy()
    negatives = df[df["_is_negative"]].copy()
    ambiguous = len(df) - len(positives) - len(negatives)

    print(f"   SIF-positive rows : {len(positives):,}")
    print(f"   Non-SIF rows      : {len(negatives):,}")
    print(f"   Ambiguous (dropped): {ambiguous:,}")

    # --- Build standardized output frames ---
    def build(frame: pd.DataFrame, label: int, src: str) -> pd.DataFrame:
        out = pd.DataFrame()
        out["Description"] = frame[TEXT_COL]
        out["is_sif"] = label
        out["source"] = src
        # Same placeholder metadata for BOTH classes, so the model cannot use
        # metadata as a shortcut to guess the label (see limitations.md).
        out["Industry Sector"] = "Others"
        out["Employee or Third Party"] = "Employee"
        out["iogp_rule"] = "Uncategorized"
        out = out.drop_duplicates(subset=["Description"]).reset_index(drop=True)
        return out

    pos_out = build(positives, 1, "OSHA_ABSTRACTS")
    neg_out = build(negatives, 0, "OSHA_ABSTRACTS")

    POS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pos_out.to_csv(POS_OUT, index=False)
    neg_out.to_csv(NEG_OUT, index=False)

    print(f"\n4. Saved {len(pos_out):,} positives -> {POS_OUT}")
    print(f"   Saved {len(neg_out):,} negatives -> {NEG_OUT}")

    # --- Show samples so you can eyeball the label quality ---
    print("\n5. Sample NEGATIVE narratives (verify these look genuinely minor):")
    for i, t in enumerate(neg_out["Description"].head(3), 1):
        print(f"\n   [{i}] {t[:260]}...")

    print("\n6. Sample POSITIVE narratives (verify these look genuinely severe):")
    for i, t in enumerate(pos_out["Description"].head(2), 1):
        print(f"\n   [{i}] {t[:260]}...")


if __name__ == "__main__":
    main()