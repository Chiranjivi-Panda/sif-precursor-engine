"""
Embedding pipeline for the SIF Precursor Engine.

Encodes the 'Description' column using a SentenceTransformer model
and caches the result to disk as a .npy file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH


def embed_descriptions(
    df: pd.DataFrame,
    cache_path: str | Path = EMBEDDINGS_PATH,
    force_recompute: bool = False,
) -> np.ndarray:
    """Embed the 'Description' column with SentenceTransformer.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Description' column.
    cache_path : str | Path
        Where to save / load the .npy embedding cache.
    force_recompute : bool
        If True, ignore any existing cache and re-encode from scratch.

    Returns
    -------
    np.ndarray
        Shape (len(df), embedding_dim).
    """
    cache_path = Path(cache_path)
    n_rows = len(df)

    # -- 1. Try loading from cache ------------------------------------
    if cache_path.exists() and not force_recompute:
        print(f"[embed] Loading cached embeddings from: {cache_path}")
        embeddings = np.load(cache_path)

        if embeddings.shape[0] != n_rows:
            raise RuntimeError(
                f"[embed] MISMATCH: cached embeddings have {embeddings.shape[0]} rows "
                f"but the dataframe has {n_rows} rows. The cache is stale or misaligned. "
                f"Re-run with force_recompute=True to regenerate embeddings."
            )

        print(f"[embed] Cached embeddings shape: {embeddings.shape}")
        print(f"[embed] [OK] Row count matches dataframe ({n_rows}).")
        return embeddings

    # -- 2. Compute fresh embeddings ----------------------------------
    print(f"[embed] No valid cache found (or force_recompute=True).")
    print(f"[embed] Loading DistilBERT model for embeddings ...")

    import torch
    from transformers import DistilBertTokenizer, DistilBertModel

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    descriptions = df["Description"].tolist()
    print(f"[embed] Encoding {len(descriptions)} descriptions ...")
    
    embeddings = []
    with torch.no_grad():
        for i, text in enumerate(descriptions):
            if i % 50 == 0:
                print(f"[embed] Processed {i}/{len(descriptions)} ...")
            inputs = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            # Extract [CLS] token's final hidden state (768-dim)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(cls_embedding)

    embeddings = np.array(embeddings)

    # -- 3. Save to cache ---------------------------------------------
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"[embed] Saved embeddings to: {cache_path}")

    # -- 4. Print shape confirmation ----------------------------------
    print(f"[embed] Embedding shape: {embeddings.shape}")
    expected_dim = 768
    if embeddings.shape != (n_rows, expected_dim):
        print(
            f"[embed] [WARN] Expected shape ({n_rows}, {expected_dim}), "
            f"got {embeddings.shape}."
        )
    else:
        print(f"[embed] [OK] Shape confirmed: ({n_rows}, {expected_dim}).")

    return embeddings
