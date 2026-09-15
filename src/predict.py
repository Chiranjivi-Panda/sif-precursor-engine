"""
Unified Inference Module for SIF Precursor Engine

Model A : SIF binary classification  -- 768-dim DistilBERT embeddings ONLY
Model B : IOGP Life-Saving Rule tag  -- 768-dim DistilBERT embeddings ONLY

DESIGN NOTE (important, see reports/limitations.md)
---------------------------------------------------
Industry Sector and Employee Type are accepted by predict_report() for UI
compatibility but are NOT used as model features. During development we found
that concatenating these one-hot features let XGBoost learn a dataset-origin
shortcut: the same hazard text scored 99% SIF with one sector selected and 31%
with another. Hazard severity must be driven by the narrative, not by metadata.
"""

import os
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import DistilBertTokenizer, DistilBertModel

from src.config import (
    PROCESSED_CSV_PATH,
    EMBEDDINGS_PATH,
    MODELS_DIR,
    EMBEDDING_MODEL_NAME,
    IOGP_MIN_CONFIDENCE,
)

SYNTH_SOURCE = "SYNTHETIC_NEG"
DEFAULT_THRESHOLD = 0.5
VERBOSE = False          # set True to print per-stage timings

_cache = {}


# ---------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------
def load_artifacts():
    """Load models, embedder, and reference corpus once."""
    print("[predict] Loading models and artifacts...")
    artifacts = {}

    def safe_load(path):
        try:
            return joblib.load(path)
        except Exception as e:
            import importlib.metadata

            def get_ver(pkg):
                try:
                    return importlib.metadata.version(pkg)
                except Exception:
                    return "Not installed"

            msg = (
                f"Failed to load artifact: {os.path.basename(path)}\n"
                f"Exception: {e}\n"
                f"Installed versions: numpy={get_ver('numpy')}, "
                f"scipy={get_ver('scipy')}, "
                f"scikit-learn={get_ver('scikit-learn')}, "
                f"xgboost={get_ver('xgboost')}\n"
                "Pickled artifacts were likely created with different library "
                "versions. Re-run src/train_model_a.py and src/train_model_b.py "
                "in this environment to regenerate them."
            )
            raise RuntimeError(msg) from e

    # ---- Model A ----
    artifacts["model_a"] = safe_load(os.path.join(MODELS_DIR, "model_a_sif_classifier.joblib"))

    prep_path = os.path.join(MODELS_DIR, "model_a_preprocessor.joblib")
    prep_a = safe_load(prep_path) if os.path.exists(prep_path) else {}

    if isinstance(prep_a, dict) and prep_a.get("mode") == "embeddings_only":
        artifacts["threshold"] = float(prep_a.get("threshold", DEFAULT_THRESHOLD))
        artifacts["expected_dim"] = int(prep_a.get("embedding_dim", 768))
    else:
        # Legacy artifact detected -- refuse to silently misbehave
        raise RuntimeError(
            "model_a_preprocessor.joblib is in the OLD format (with one-hot "
            "encoders). Model A has been retrained on embeddings only.\n"
            "Fix: run  python src/train_model_a.py  to regenerate artifacts."
        )

    # ---- Model B ----
    artifacts["model_b"] = safe_load(os.path.join(MODELS_DIR, "model_b_iogp_classifier.joblib"))
    with open(os.path.join(MODELS_DIR, "model_b_classes.pkl"), "rb") as f:
        artifacts["classes_b"] = pickle.load(f)

    # ---- Embedder ----
    artifacts["tokenizer"] = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts["model"] = bert.to(device).eval()
    artifacts["device"] = device

    # ---- Reference corpus for nearest-neighbour explanations ----
    train_emb = np.load(EMBEDDINGS_PATH)
    train_df = pd.read_csv(PROCESSED_CSV_PATH)

    # Synthetic rows are training aids, not real incidents. Never show them
    # to a user as a "similar historical report".
    if "source" in train_df.columns:
        real_mask = (train_df["source"] != SYNTH_SOURCE).to_numpy()
    else:
        real_mask = np.ones(len(train_df), dtype=bool)

    artifacts["nn_embeddings"] = train_emb[real_mask]
    artifacts["nn_df"] = train_df[real_mask].reset_index(drop=True)

    # Backward-compatible aliases for precompute_scores.py and legacy scripts.
    # These point at the REAL-ONLY rows on purpose: the dashboard must never
    # display synthetic training rows as if they were genuine incidents.
    artifacts["train_embeddings"] = artifacts["nn_embeddings"]
    artifacts["train_df"] = artifacts["nn_df"]

    n_hidden = int((~real_mask).sum())
    print(f"[predict] Threshold = {artifacts['threshold']:.2f} "
          f"| neighbour pool = {len(artifacts['nn_df']):,} real rows "
          f"({n_hidden:,} synthetic excluded)")
    print("[predict] All artifacts loaded successfully.")
    return artifacts


def set_artifacts(artifacts):
    global _cache
    _cache.clear()
    _cache.update(artifacts)


# ---------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------
def _embed(text: str) -> np.ndarray:
    tokenizer = _cache["tokenizer"]
    bert = _cache["model"]
    device = _cache["device"]
    with torch.no_grad():
        inputs = tokenizer(str(text), return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        out = bert(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().numpy()   # (1, 768) CLS


# ---------------------------------------------------------
# Core inference
# ---------------------------------------------------------
def predict_report(text: str,
                   industry_sector: str = None,
                   employee_type: str = None) -> dict:
    """
    Classify a free-text safety report.

    industry_sector / employee_type are accepted for UI compatibility but are
    intentionally NOT used as model features (see module docstring).
    """
    if not _cache:
        set_artifacts(load_artifacts())

    t0 = time.time()

    # 1) Embed
    emb = _embed(text)
    t1 = time.time()

    # 2) Model A -- embeddings only, tuned threshold
    model_a = _cache["model_a"]
    threshold = _cache["threshold"]

    assert emb.shape[1] == _cache["expected_dim"], (
        f"Embedding dim mismatch: model expects {_cache['expected_dim']}, "
        f"got {emb.shape[1]}"
    )

    sif_probability = float(model_a.predict_proba(emb)[0][1])
    is_sif = bool(sif_probability >= threshold)
    t2 = time.time()

    # 3) Model B -- only when flagged SIF
    
    iogp_rule = None
    iogp_confidence = None
    iogp_uncertain = False

    if is_sif:
        probs = _cache["model_b"].predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        # Softmax always sums to 1, so a "confident" score can still come from
        # a class the model never learned. Suppress tags below the floor
        # rather than present a wrong rule authoritatively.
        if conf >= IOGP_MIN_CONFIDENCE:
            iogp_rule = str(_cache["classes_b"][idx])
            iogp_confidence = conf
        else:
            iogp_rule = "Uncertain -- manual review required"
            iogp_confidence = conf
            iogp_uncertain = True
    t3 = time.time()
    # 4) Nearest real historical reports
    sims = cosine_similarity(emb, _cache["nn_embeddings"])[0]
    top3 = np.argsort(sims)[-3:][::-1]
    nn_df = _cache["nn_df"]

    nearest_neighbors = []
    for i in top3:
        row = nn_df.iloc[int(i)]
        nearest_neighbors.append({
            "text": str(row["Description"]),
            "is_sif": int(row["is_sif"]),
            "iogp_rule": str(row.get("iogp_rule", "Uncategorized")),
            "similarity": float(sims[int(i)]),
            "source": str(row.get("source", "unknown")),
        })

    if VERBOSE:
        print(f"[predict] embed={t1-t0:.3f}s  modelA={t2-t1:.3f}s  "
              f"modelB={t3-t2:.3f}s  nn={time.time()-t3:.3f}s  "
              f"total={time.time()-t0:.3f}s")

    return {
        "is_sif": is_sif,
        "sif_probability": sif_probability,
        "threshold": threshold,
        "iogp_rule": iogp_rule,
        "iogp_confidence": iogp_confidence,
        "nearest_neighbors": nearest_neighbors,        
        "iogp_uncertain": iogp_uncertain,
    }