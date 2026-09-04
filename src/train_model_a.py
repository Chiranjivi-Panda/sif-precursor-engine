"""
Model A Training Script: Binary SIF Classification

This script predicts the `is_sif` target (Serious Injury or Fatality) using two approaches:
1. TF-IDF + Logistic Regression (Baseline)
2. MiniLM Embeddings + XGBoost (Proposed)

EXCLUDED FEATURES (Label Leakage / Ethics):
- 'Critical Risk', 'Accident Level', 'Potential Accident Level', 'Genre'
These must NOT be used because they either directly encode the target label,
or are unethical/demographic variables ('Genre'/Gender).

Input Features:
- Cached description embeddings (384-dim)
- One-hot encoded 'Industry Sector' and 'Employee or Third Party'
"""

import os
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from xgboost import XGBClassifier

from config import PROCESSED_CSV_PATH, EMBEDDINGS_PATH, MODELS_DIR, RANDOM_STATE, PROJECT_ROOT

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure output directories exist
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print("  Model A: SIF Classification (Binary)")
    print("=" * 65)

    # 1. Load Data
    print(f"Loading data from {PROCESSED_CSV_PATH} ...")
    df = pd.read_csv(PROCESSED_CSV_PATH)
    
    print(f"Loading embeddings from {EMBEDDINGS_PATH} ...")
    embeddings = np.load(EMBEDDINGS_PATH)
    
    assert len(df) == embeddings.shape[0], "Mismatch between dataframe rows and embeddings."
    
    # 2. Extract Labels and Features
    y = df['is_sif'].values
    descriptions = df['Description'].fillna("").values
    
    # One-hot encode categorical features
    # Excluded: Critical Risk, Accident Level, Potential Accident Level, Genre
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_features = df[['Industry Sector', 'Employee or Third Party']]
    cat_encoded = ohe.fit_transform(cat_features)
    
    # Combine embeddings with categorical features
    X_embed = np.hstack([embeddings, cat_encoded])
    
    expected_features = 768 + cat_encoded.shape[1]
    assert X_embed.shape[1] == expected_features, f"Expected {expected_features} features, got {X_embed.shape[1]}"
    
    # 3. Setup StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Storage for metrics
    metrics = {
        'TFIDF + LogReg': {'precision': [], 'recall': [], 'f1': [], 'f2': []},
        'MiniLM + XGBoost': {'precision': [], 'recall': [], 'f1': [], 'f2': []}
    }
    
    # Storage for representative fold (Fold 0) confusion matrix & examples
    conf_matrix_rep = None
    rep_predictions = None
    
    print("\nStarting 5-Fold Cross-Validation...")
    
    fold = 0
    for train_idx, test_idx in skf.split(X_embed, y):
        # Data splits
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- Approach 1: TFIDF + LogReg ---
        desc_train, desc_test = descriptions[train_idx], descriptions[test_idx]
        
        tfidf = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf.fit_transform(desc_train)
        X_test_tfidf = tfidf.transform(desc_test)
        
        lr_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)
        lr_model.fit(X_train_tfidf, y_train)
        y_pred_lr = lr_model.predict(X_test_tfidf)
        
        metrics['TFIDF + LogReg']['precision'].append(precision_score(y_test, y_pred_lr, zero_division=0))
        metrics['TFIDF + LogReg']['recall'].append(recall_score(y_test, y_pred_lr, zero_division=0))
        metrics['TFIDF + LogReg']['f1'].append(f1_score(y_test, y_pred_lr, zero_division=0))
        metrics['TFIDF + LogReg']['f2'].append(fbeta_score(y_test, y_pred_lr, beta=2, zero_division=0))
        
        # --- Approach 2: MiniLM + XGBoost ---
        X_train_xgb, X_test_xgb = X_embed[train_idx], X_embed[test_idx]
        
        # Compute scale_pos_weight dynamically per fold
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        xgb_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_xgb, y_train)
        
        # Threshold tuning deferred to enhancement stage; tuning on the same data as final reported metrics would overfit.
        # We use default 0.5 implicitly via .predict()
        y_pred_xgb = xgb_model.predict(X_test_xgb)
        y_prob_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
        
        metrics['MiniLM + XGBoost']['precision'].append(precision_score(y_test, y_pred_xgb, zero_division=0))
        metrics['MiniLM + XGBoost']['recall'].append(recall_score(y_test, y_pred_xgb, zero_division=0))
        metrics['MiniLM + XGBoost']['f1'].append(f1_score(y_test, y_pred_xgb, zero_division=0))
        metrics['MiniLM + XGBoost']['f2'].append(fbeta_score(y_test, y_pred_xgb, beta=2, zero_division=0))
        
        # Capture fold 0 for detailed analysis
        if fold == 0:
            conf_matrix_rep = confusion_matrix(y_test, y_pred_xgb)
            rep_predictions = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred_xgb,
                'probability': y_prob_xgb,
                'description': desc_test
            })
            
        fold += 1

    # 4. Report Metrics
    print("\n" + "=" * 85)
    print(f"| {'Approach':<21} | {'Precision':<12} | {'Recall':<12} | {'F1':<12} | {'F2':<12} |")
    print("-" * 85)
    
    for approach, res in metrics.items():
        p_m, p_s = np.mean(res['precision']), np.std(res['precision'])
        r_m, r_s = np.mean(res['recall']), np.std(res['recall'])
        f1_m, f1_s = np.mean(res['f1']), np.std(res['f1'])
        f2_m, f2_s = np.mean(res['f2']), np.std(res['f2'])
        
        print(f"| {approach:<21} | {p_m:.2f} ± {p_s:.2f} | {r_m:.2f} ± {r_s:.2f} | {f1_m:.2f} ± {f1_s:.2f} | {f2_m:.2f} ± {f2_s:.2f} |")
    print("=" * 85)
    
    # Leakage check
    xgb_f2 = np.mean(metrics['MiniLM + XGBoost']['f2'])
    if xgb_f2 > 0.95:
        print("\n🚨 WARNING: F2 score is suspiciously high (> 0.95). Potential label leakage detected!")
    
    # 5. Output Additional Analysis (Fold 0)
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_rep, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-SIF (0)', 'SIF (1)'], yticklabels=['Non-SIF (0)', 'SIF (1)'])
    plt.title("Confusion Matrix - MiniLM + XGBoost (Fold 0)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = REPORTS_DIR / "model_a_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"\nSaved representative confusion matrix to: {cm_path}")
    
    # Examples
    print("\n--- Example Predictions (Fold 0) ---")
    
    tp = rep_predictions[(rep_predictions['actual'] == 1) & (rep_predictions['predicted'] == 1)]
    fp = rep_predictions[(rep_predictions['actual'] == 0) & (rep_predictions['predicted'] == 1)]
    fn = rep_predictions[(rep_predictions['actual'] == 1) & (rep_predictions['predicted'] == 0)]
    
    def print_example(name, df_subset):
        if len(df_subset) > 0:
            ex = df_subset.iloc[0]
            desc_short = ex['description'][:200].replace('\n', ' ')
            if len(ex['description']) > 200: desc_short += "..."
            print(f"\n[{name}]")
            print(f"Text: \"{desc_short}\"")
            print(f"Actual: {ex['actual']} | Predicted: {ex['predicted']} | Prob: {ex['probability']:.4f}")
        else:
            print(f"\n[{name}] No examples found in this fold.")

    print_example("True Positive", tp)
    print_example("False Positive", fp)
    print_example("False Negative", fn)
    
    # 6. Final Model Training on Full Data
    print("\nTraining final production model on full dataset...")
    neg_count_full = (y == 0).sum()
    pos_count_full = (y == 1).sum()
    scale_pos_weight_full = neg_count_full / pos_count_full if pos_count_full > 0 else 1.0
    
    final_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight_full,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    final_model.fit(X_embed, y)
    
    # Save artifacts
    model_path = MODELS_DIR / "model_a_sif_classifier.joblib"
    preprocessor_path = MODELS_DIR / "model_a_preprocessor.joblib"
    
    joblib.dump(final_model, model_path)
    joblib.dump({'ohe': ohe, 'expected_features': expected_features, 'cat_cols': ['Industry Sector', 'Employee or Third Party']}, preprocessor_path)
    
    print(f"Saved final model to: {model_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
