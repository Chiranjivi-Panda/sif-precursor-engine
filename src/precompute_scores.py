import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- existing imports below ----
import time
import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_CSV_PATH, EMBEDDINGS_PATH
from src.predict import load_artifacts, set_artifacts, _cache

"""
src/precompute_scores.py

Scores every REAL report in the corpus with Model A + Model B and writes
data/processed/scored_reports.csv for the dashboard.

Synthetic training rows are excluded -- dashboard analytics must only ever
reflect genuine incidents.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_CSV_PATH, EMBEDDINGS_PATH
from src.predict import load_artifacts, set_artifacts, _cache

SYNTH = "SYNTHETIC_NEG"
OUT = Path("data/processed/scored_reports.csv")


def main():
    print("=" * 60)
    print("  Precomputing Dashboard Scores")
    print("=" * 60)

    set_artifacts(load_artifacts())

    df = pd.read_csv(PROCESSED_CSV_PATH)
    emb = np.load(EMBEDDINGS_PATH)
    assert len(df) == len(emb), f"Row mismatch: {len(df)} vs {len(emb)}"

    # ---- real rows only ----
    if "source" in df.columns:
        mask = (df["source"] != SYNTH).to_numpy()
    else:
        mask = np.ones(len(df), dtype=bool)

    df = df[mask].reset_index(drop=True)
    emb = emb[mask]
    print(f"\nScoring {len(df):,} real rows ({int((~mask).sum()):,} synthetic skipped)")

    model_a = _cache["model_a"]
    model_b = _cache["model_b"]
    classes_b = _cache["classes_b"]
    threshold = _cache["threshold"]

    t0 = time.time()

    # ---- Model A (batch) ----
    probs = model_a.predict_proba(emb)[:, 1]
    preds = (probs >= threshold).astype(int)

    # ---- Model B (batch, SIF rows only) ----
    iogp_rule = np.array([None] * len(df), dtype=object)
    iogp_conf = np.full(len(df), np.nan)

    sif_rows = np.where(preds == 1)[0]
    if len(sif_rows):
        pb = model_b.predict_proba(emb[sif_rows])
        idx = pb.argmax(axis=1)
        iogp_rule[sif_rows] = [str(classes_b[i]) for i in idx]
        iogp_conf[sif_rows] = pb.max(axis=1)

    df["sif_probability"] = probs
    df["predicted_is_sif"] = preds
    df["predicted_iogp_rule"] = iogp_rule
    df["iogp_confidence"] = iogp_conf

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("\n[Summary]")
    print(f"  Rows scored     : {len(df):,}")
    print(f"  Threshold       : {threshold:.2f}")
    print(f"  Predicted SIF   : {int(preds.sum()):,} ({preds.mean()*100:.1f}%)")
    if "is_sif" in df.columns:
        acc = (preds == df["is_sif"].to_numpy()).mean()
        print(f"  Agreement w/ label: {acc*100:.1f}%")
    print(f"  Mean SIF prob   : {probs.mean():.4f}")
    print(f"  Elapsed         : {time.time()-t0:.2f}s")
    print(f"  Saved to        : {OUT}")


if __name__ == "__main__":
    main()