"""
Model B Training Script: IOGP Rule Classification (Multi-class)

This script predicts the `iogp_rule` target.
Classes with < 10 samples are explicitly warned about.

EXCLUDED FEATURES (Label Leakage / Ethics):
- 'Industry Sector', 'Employee or Third Party' (excluded for this model per spec)
- 'Critical Risk', 'Accident Level', 'Potential Accident Level', 'Genre'
Critical Risk is literally the source of the iogp_rule label — including it would be direct label leakage.

Input Features:
- Cached description embeddings (384-dim) ONLY.
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import PROCESSED_CSV_PATH, EMBEDDINGS_PATH, MODELS_DIR, RANDOM_STATE, PROJECT_ROOT

warnings.filterwarnings("ignore", category=UserWarning)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print("  Model B: IOGP Rule Classification (Multi-class)")
    print("=" * 65)

    # 1. Load Data
    df = pd.read_csv(PROCESSED_CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    
    assert len(df) == embeddings.shape[0], "Mismatch between dataframe rows and embeddings."
    
    # 2. Data Filtering
    # Exclude all rows where iogp_rule == "Uncategorized"
    valid_idx = df['iogp_rule'] != "Uncategorized"
    df_filtered = df[valid_idx].copy()
    X = embeddings[valid_idx]
    y_str = df_filtered['iogp_rule'].values
    
    print(f"\nFiltered out 'Uncategorized'. Remaining rows: {len(df_filtered)}")
    
    class_counts = df_filtered['iogp_rule'].value_counts()
    print("\nPer-class counts for each IOGP rule:")
    for rule, count in class_counts.items():
        print(f"  - {rule}: {count}")
        if count < 10:
            print(f"    [WARNING] Class '{rule}' has fewer than 10 examples. Metrics will be unreliable.")
            
    # Encode targets to 0..N-1 for XGBoost
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = le.classes_
    
    # 3. Setup Validation Strategy
    # Using 3 folds instead of 5 due to small per-class sample sizes
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    metrics = {
        'LogReg (multinomial)': [],
        'XGBoost (multi)': []
    }
    
    conf_matrix_rep = None
    clf_report_rep = None
    y_test_rep = None
    y_pred_rep = None
    best_model_name = None
    
    print("\nStarting 3-Fold Cross-Validation...")
    
    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- Model 1: Logistic Regression ---
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        # --- Model 2: XGBoost ---
        xgb = XGBClassifier(
            objective='multi:softprob',
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        
        # Macro F1
        metrics['LogReg (multinomial)'].append(f1_score(y_test, y_pred_lr, average='macro', zero_division=0))
        metrics['XGBoost (multi)'].append(f1_score(y_test, y_pred_xgb, average='macro', zero_division=0))
        
        # Save Fold 0 data for representative metrics
        # (We use LogReg if it's Fold 0, but we don't know the winner yet. 
        # Actually, let's just save the test set and we can compute it for the winner later, 
        # or we just pick LogReg or XGBoost to show in fold 0.)
        if fold == 0:
            # We'll just store the predictions for both and pick the winner's predictions at the end
            fold0_data = {
                'y_test': y_test,
                'y_pred_lr': y_pred_lr,
                'y_pred_xgb': y_pred_xgb
            }
        fold += 1

    # 4. Report Metrics
    lr_mean, lr_std = np.mean(metrics['LogReg (multinomial)']), np.std(metrics['LogReg (multinomial)'])
    xgb_mean, xgb_std = np.mean(metrics['XGBoost (multi)']), np.std(metrics['XGBoost (multi)'])
    
    print("\n" + "=" * 45)
    print(f"| {'Approach':<21} | {'Macro-F1':<17} |")
    print("-" * 45)
    print(f"| {'LogReg (multinomial)':<21} | {lr_mean:.2f} ± {lr_std:.2f}     |")
    print(f"| {'XGBoost (multi)':<21} | {xgb_mean:.2f} ± {xgb_std:.2f}     |")
    print("=" * 45)
    
    # 5. Determine Winner and Output Fold 0 Details
    winner_name = 'LogReg (multinomial)' if lr_mean > xgb_mean else 'XGBoost (multi)'
    print(f"\nWinner selected based on Macro-F1: {winner_name}")
    
    # Representative metrics from Fold 0 for the winner
    y_test_rep = fold0_data['y_test']
    y_pred_rep = fold0_data['y_pred_lr'] if winner_name == 'LogReg (multinomial)' else fold0_data['y_pred_xgb']
    
    conf_matrix_rep = confusion_matrix(y_test_rep, y_pred_rep)
    clf_report_rep = classification_report(y_test_rep, y_pred_rep, target_names=class_names, zero_division=0)
    
    print("\n--- Per-Class Metrics Table (Fold 0, Winner Model) ---")
    print(clf_report_rep)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_rep, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {winner_name} (Fold 0)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    cm_path = REPORTS_DIR / "model_b_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"Saved representative confusion matrix to: {cm_path}")
    
    # 6. Train Final Production Model
    print("\nTraining final production model on full filtered dataset...")
    if winner_name == 'LogReg (multinomial)':
        final_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    else:
        final_model = XGBClassifier(
            objective='multi:softprob',
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
    final_model.fit(X, y)
    
    model_path = MODELS_DIR / "model_b_iogp_classifier.joblib"
    classes_path = MODELS_DIR / "model_b_classes.pkl"
    
    joblib.dump(final_model, model_path)
    with open(classes_path, 'wb') as f:
        pickle.dump(class_names, f)
        
    print(f"Saved final model to: {model_path}")
    print(f"Saved class labels to: {classes_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
