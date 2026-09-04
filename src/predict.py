"""
Unified Inference Module for SIF Precursor Engine

Provides the `predict_report` function which combines Model A (SIF prediction)
and Model B (IOGP rule multi-class prediction) for dashboard use.
Models and artifacts are loaded once and cached at the module level.
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
    EMBEDDING_MODEL_NAME
)

# ---------------------------------------------------------
# Module-level Cache for Models and Artifacts
# ---------------------------------------------------------
_cache = {}

def _load_artifacts():
    """Load all models, embedders, and reference data once."""
    if _cache:
        return
    
    print("[predict] Loading models and artifacts...")
    
    # Model A (SIF Classification)
    model_a_path = MODELS_DIR / "model_a_sif_classifier.joblib"
    prep_a_path = MODELS_DIR / "model_a_preprocessor.joblib"
    _cache['model_a'] = joblib.load(model_a_path)
    _cache['prep_a'] = joblib.load(prep_a_path)
    
    # Model B (IOGP Rule Classification)
    model_b_path = MODELS_DIR / "model_b_iogp_classifier.joblib"
    classes_b_path = MODELS_DIR / "model_b_classes.pkl"
    _cache['model_b'] = joblib.load(model_b_path)
    with open(classes_b_path, 'rb') as f:
        _cache['classes_b'] = pickle.load(f)
        
    # Embedder
    _cache['tokenizer'] = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    _cache['model'] = model
    _cache['device'] = device
    # Training Data (for nearest neighbor lookup)
    _cache['train_embeddings'] = np.load(EMBEDDINGS_PATH)
    _cache['train_df'] = pd.read_csv(PROCESSED_CSV_PATH)
    
    print("[predict] All artifacts loaded successfully.")

# ---------------------------------------------------------
# Core Inference Function
# ---------------------------------------------------------
def predict_report(text: str, industry_sector: str = None, employee_type: str = None) -> dict:
    """
    Predicts whether a safety report is a SIF precursor (Model A) and, 
    if so, identifies the IOGP Life-Saving Rule (Model B).
    
    Also returns the top 3 most similar historical reports.
    """
    _load_artifacts()
    
    start_time = time.time()
    
    # 1. Embedding
    tokenizer = _cache['tokenizer']
    model = _cache['model']
    device = _cache['device']
    
    with torch.no_grad():
        inputs = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        # Extract [CLS] token's final hidden state (768-dim)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    emb = cls_embedding  # Shape: (1, 768)
    
    t_emb = time.time()
    print(f"Embedding time: {t_emb - start_time:.4f}s")
    
    # 2. Feature Engineering for Model A
    prep_a = _cache['prep_a']
    ohe = prep_a['ohe']
    cat_cols = prep_a['cat_cols']
    
    # Create 1-row DataFrame for categorical features
    cat_df = pd.DataFrame([{
        'Industry Sector': industry_sector if industry_sector else 'Unknown',
        'Employee or Third Party': employee_type if employee_type else 'Unknown'
    }])
    
    cat_encoded = ohe.transform(cat_df)
    
    # Concatenate features
    X_a = np.hstack([emb, cat_encoded])
    
    # Assertion
    assert X_a.shape[1] == prep_a['expected_features'], (
        f"Feature mismatch: Expected {prep_a['expected_features']}, got {X_a.shape[1]}"
    )
    
    # 3. Model A Prediction (SIF)
    model_a = _cache['model_a']
    # Threshold fixed at 0.5 as per training
    sif_probability = model_a.predict_proba(X_a)[0][1]
    is_sif = sif_probability >= 0.5
    
    t_ma = time.time()
    print(f"Model A time: {t_ma - t_emb:.4f}s")
    
    # 4. Model B Prediction (IOGP Rule)
    # We only tag IOGP rules for SIF-flagged reports; presenting a confident 
    # IOGP prediction for a non-SIF report would be misleading.
    iogp_rule = None
    iogp_confidence = None
    
    if is_sif:
        model_b = _cache['model_b']
        classes_b = _cache['classes_b']
        
        probs = model_b.predict_proba(emb)[0]
        pred_idx = np.argmax(probs)
        
        iogp_rule = classes_b[pred_idx]
        iogp_confidence = probs[pred_idx]
        
    t_mb = time.time()
    print(f"Model B time: {t_mb - t_ma:.4f}s")
        
    # 5. Nearest Neighbors Lookup
    train_embeddings = _cache['train_embeddings']
    train_df = _cache['train_df']
    
    similarities = cosine_similarity(emb, train_embeddings)[0]
    # Get top 3 indices (argsort sorts ascending, so take last 3 and reverse)
    top3_indices = np.argsort(similarities)[-3:][::-1]
    
    nearest_neighbors = []
    for idx in top3_indices:
        row = train_df.iloc[idx]
        nearest_neighbors.append({
            'text': row['Description'],
            'is_sif': int(row['is_sif']),
            'iogp_rule': str(row['iogp_rule'])
        })
        
    t_nn = time.time()
    print(f"Nearest neighbors time: {t_nn - t_mb:.4f}s")
    print(f"Total prediction time: {time.time() - start_time:.4f}s")
        
    # 6. Return Formatting
    return {
        'is_sif': bool(is_sif),
        'sif_probability': float(sif_probability),
        'iogp_rule': str(iogp_rule) if iogp_rule else None,
        'iogp_confidence': float(iogp_confidence) if iogp_confidence else None,
        'nearest_neighbors': nearest_neighbors
    }
