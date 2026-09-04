#!/usr/bin/env python
"""
Standalone check: embed descriptions and verify row-order alignment.

1. Loads labeled_data.csv.
2. Runs embed_descriptions (uses cache if available).
3. Sanity-checks that embedding row order matches dataframe row order
   by re-embedding the first 3 descriptions independently and confirming
   cosine similarity to the cached versions is ~1.0.

Usage:
    python src/run_embed.py
"""

import numpy as np
import pandas as pd

from config import PROCESSED_CSV_PATH, EMBEDDING_MODEL_NAME
from embed import embed_descriptions


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def main() -> None:
    print("=" * 65)
    print("  SIF Precursor Engine -- Embedding Check")
    print("=" * 65)

    # -- Load processed data ------------------------------------------
    print(f"\n[run_embed] Loading processed data from: {PROCESSED_CSV_PATH}")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    print(f"[run_embed] DataFrame shape: {df.shape}")

    # -- Embed (or load from cache) -----------------------------------
    embeddings = embed_descriptions(df)
    print(f"\n[run_embed] Final embedding shape: {embeddings.shape}")

    # -- Sanity check: row-order alignment ----------------------------
    # Re-embed the first 3 descriptions from scratch (no cache) and
    # compare against the corresponding rows in the full embedding.
    print("\n" + "-" * 65)
    print("  Row-order alignment sanity check (first 3 rows)")
    print("-" * 65)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    first_3 = df["Description"].head(3).tolist()
    fresh_embeddings = model.encode(first_3, show_progress_bar=False)
    fresh_embeddings = np.array(fresh_embeddings)

    all_ok = True
    for i in range(3):
        sim = _cosine_similarity(embeddings[i], fresh_embeddings[i])
        status = "[OK]" if sim > 0.999 else "[FAIL]"
        if sim <= 0.999:
            all_ok = False
        desc_preview = first_3[i][:60] + "..." if len(first_3[i]) > 60 else first_3[i]
        print(f"  Row {i}: cosine_sim = {sim:.6f}  {status}")
        print(f"          \"{desc_preview}\"")

    print("-" * 65)
    if all_ok:
        print("  [OK] All 3 rows match -- embeddings are correctly aligned.")
    else:
        print("  [FAIL] Mismatch detected! Cache may be stale or row-shuffled.")
        print("         Re-run embed_descriptions with force_recompute=True.")
    print("=" * 65)


if __name__ == "__main__":
    main()
