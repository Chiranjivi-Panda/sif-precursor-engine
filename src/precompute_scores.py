import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_artifacts, set_artifacts, _cache
from src.config import PROJECT_ROOT, PROCESSED_CSV_PATH

def main():
    print("=" * 60)
    print("  Precomputing Dashboard Scores")
    print("=" * 60)
    
    start_time = time.time()
    
    set_artifacts(load_artifacts())
    
    df = _cache['train_df']
    embeddings = _cache['train_embeddings']
    prep_a = _cache['prep_a']
    model_a = _cache['model_a']
    model_b = _cache['model_b']
    classes_b = _cache['classes_b']
    
    results = []
    total = len(df)
    pred_sif_count = 0
    sif_probs = []
    
    for i in range(total):
        row = df.iloc[i]
        emb = embeddings[i].reshape(1, -1)
        
        ind = row.get('Industry Sector', 'Unknown')
        emp = row.get('Employee or Third Party', 'Unknown')
        
        cat_df = pd.DataFrame([{
            'Industry Sector': ind if pd.notna(ind) else 'Unknown',
            'Employee or Third Party': emp if pd.notna(emp) else 'Unknown'
        }])
        
        cat_encoded = prep_a['ohe'].transform(cat_df)
        X_a = np.hstack([emb, cat_encoded])
        
        sif_prob = float(model_a.predict_proba(X_a)[0][1])
        is_sif = sif_prob >= 0.5
        
        pred_iogp = ""
        iogp_conf = ""
        
        if is_sif:
            probs = model_b.predict_proba(emb)[0]
            pred_idx = np.argmax(probs)
            pred_iogp = classes_b[pred_idx]
            iogp_conf = float(probs[pred_idx])
            pred_sif_count += 1
            
        sif_probs.append(sif_prob)
            
        results.append({
            'original_index': i,
            'Description': row.get('Description', ''),
            'Local': row.get('Local', ''),
            'Industry Sector': ind,
            'is_sif': row.get('is_sif', ''),
            'iogp_rule': row.get('iogp_rule', ''),
            'pred_sif': int(is_sif),
            'sif_probability': sif_prob,
            'pred_iogp_rule': pred_iogp,
            'iogp_confidence': iogp_conf if iogp_conf != "" else None
        })
        
    out_df = pd.DataFrame(results)
    out_path = os.path.join(PROJECT_ROOT, "data", "processed", "scored_reports.csv")
    out_df.to_csv(out_path, index=False)
    
    elapsed = time.time() - start_time
    
    print("\n[Summary]")
    print(f"Total rows scored : {total}")
    print(f"Predicted SIF     : {pred_sif_count}")
    print(f"Mean SIF Prob     : {np.mean(sif_probs):.4f}")
    print(f"Elapsed Time      : {elapsed:.2f}s")
    print(f"Saved to          : {out_path}")

if __name__ == "__main__":
    main()
