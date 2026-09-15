"""
Model B Training Script: IOGP Rule Classification (Multi-class)

Predicts the `iogp_rule` target from description embeddings.

CLASS CONSOLIDATION
-------------------
Under-represented classes are merged via IOGP_MERGE (see config.py) before
training. 'Safe Mechanical Lifting' (n=8) and 'Driving/Line of Fire' (n=9)
could not be learned -- the former scored F1 = 0.00 in CV while still being
predicted at 0.74 confidence on live input. Both are merged into
'Line of Fire', giving 4 classes that each clear 20 examples.

EXCLUDED FEATURES (Label Leakage / Ethics):
- 'Critical Risk'  -- literally the source of the iogp_rule label
- 'Accident Level', 'Potential Accident Level'  -- severity leakage
- 'Genre'  -- gender must not be a predictive input
- 'Industry Sector', 'Employee or Third Party'  -- metadata shortcut risk

Input Features:
- Cached DistilBERT description embeddings (768-dim) ONLY.
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

from config import (
    PROCESSED_CSV_PATH,
    EMBEDDINGS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    PROJECT_ROOT,
    IOGP_MERGE,
)

warnings.filterwarnings("ignore", category=UserWarning)

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("=" * 65)
    print("  Model B: IOGP Rule Classification (Multi-class)")
    print("=" * 65)

    # 1. Load Data
    df = pd.read_csv(PROCESSED_CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    assert len(df) == embeddings.shape[0], (
        f"Mismatch: {len(df)} dataframe rows vs {embeddings.shape[0]} embeddings. "
        f"Re-run  python src/run_embed.py"
    )

    # 2. Merge under-represented classes BEFORE filtering
    counts_before = df["iogp_rule"].value_counts()
    df["iogp_rule"] = df["iogp_rule"].replace(IOGP_MERGE)

    merged_any = [k for k in IOGP_MERGE if k in counts_before.index]
    if merged_any:
        print("\nClass consolidation applied (see IOGP_MERGE in config.py):")
        for src_cls in merged_any:
            print(f"  '{src_cls}' (n={counts_before[src_cls]})"
                  f"  ->  '{IOGP_MERGE[src_cls]}'")

    # 3. Data Filtering -- exclude 'Uncategorized'
    valid_idx = df["iogp_rule"] != "Uncategorized"
    df_filtered = df[valid_idx].copy()
    X = embeddings[valid_idx.to_numpy()]
    y_str = df_filtered["iogp_rule"].values

    print(f"\nFiltered out 'Uncategorized'. Remaining rows: {len(df_filtered)}")

    class_counts = df_filtered["iogp_rule"].value_counts()
    print("\nPer-class counts after consolidation:")
    for rule, count in class_counts.items():
        print(f"  - {rule}: {count}")
        if count < 10:
            print(f"    [WARNING] Class '{rule}' has fewer than 10 examples. "
                  f"Metrics will be unreliable.")

    # Encode targets to 0..N-1 for XGBoost
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = le.classes_
    print(f"\nTraining {len(class_names)} classes: {list(class_names)}")

    # 4. Setup Validation Strategy
    # 3 folds rather than 5 due to small per-class sample sizes
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    metrics = {
        "LogReg (multinomial)": [],
        "XGBoost (multi)": [],
    }

    fold0_data = None
    print("\nStarting 3-Fold Cross-Validation...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Model 1: Logistic Regression ---
        lr = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            C=1.0,
        )
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # --- Model 2: XGBoost ---
        # Reduced capacity: ~110 training rows cannot support deep trees.
        xgb = XGBClassifier(
            objective="multi:softprob",
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
            tree_method="hist",
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        f1_lr = f1_score(y_test, y_pred_lr, average="macro", zero_division=0)
        f1_xgb = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)

        metrics["LogReg (multinomial)"].append(f1_lr)
        metrics["XGBoost (multi)"].append(f1_xgb)

        print(f"  fold {fold}: LogReg macro-F1 = {f1_lr:.3f}  |  "
              f"XGBoost macro-F1 = {f1_xgb:.3f}")

        if fold == 0:
            fold0_data = {
                "y_test": y_test,
                "y_pred_lr": y_pred_lr,
                "y_pred_xgb": y_pred_xgb,
            }

    # 5. Report Metrics
    lr_mean = np.mean(metrics["LogReg (multinomial)"])
    lr_std = np.std(metrics["LogReg (multinomial)"])
    xgb_mean = np.mean(metrics["XGBoost (multi)"])
    xgb_std = np.std(metrics["XGBoost (multi)"])

    print("\n" + "=" * 45)
    print(f"| {'Approach':<21} | {'Macro-F1':<17} |")
    print("-" * 45)
    print(f"| {'LogReg (multinomial)':<21} | {lr_mean:.2f} ± {lr_std:.2f}     |")
    print(f"| {'XGBoost (multi)':<21} | {xgb_mean:.2f} ± {xgb_std:.2f}     |")
    print("=" * 45)

    # 6. Determine Winner
    winner_name = ("LogReg (multinomial)" if lr_mean > xgb_mean
                   else "XGBoost (multi)")
    print(f"\nWinner selected based on Macro-F1: {winner_name}")

    y_test_rep = fold0_data["y_test"]
    y_pred_rep = (fold0_data["y_pred_lr"]
                  if winner_name == "LogReg (multinomial)"
                  else fold0_data["y_pred_xgb"])

    labels_idx = np.arange(len(class_names))
    conf_matrix_rep = confusion_matrix(y_test_rep, y_pred_rep, labels=labels_idx)
    clf_report_rep = classification_report(
        y_test_rep, y_pred_rep,
        labels=labels_idx,
        target_names=class_names,
        zero_division=0,
    )

    print("\n--- Per-Class Metrics Table (Fold 0, Winner Model) ---")
    print(clf_report_rep)

    zero_f1 = [
        class_names[i] for i in labels_idx
        if f1_score(y_test_rep == i, y_pred_rep == i, zero_division=0) == 0.0
        and (y_test_rep == i).sum() > 0
    ]
    if zero_f1:
        print(f"[WARNING] Classes with F1 = 0.00 in fold 0: {zero_f1}")
        print("          Consider merging these in IOGP_MERGE (config.py).")
    else:
        print("[OK] No class scored F1 = 0.00 in fold 0.")

    # 7. Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        conf_matrix_rep, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {winner_name} (Fold 0)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    cm_path = os.path.join(REPORTS_DIR, "model_b_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"\nSaved representative confusion matrix to: {cm_path}")

    # 8. Train Final Production Model
    print("\nTraining final production model on full filtered dataset...")
    if winner_name == "LogReg (multinomial)":
        final_model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            C=1.0,
        )
    else:
        final_model = XGBClassifier(
            objective="multi:softprob",
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
            tree_method="hist",
        )

    final_model.fit(X, y)

    model_path = os.path.join(MODELS_DIR, "model_b_iogp_classifier.joblib")
    classes_path = os.path.join(MODELS_DIR, "model_b_classes.pkl")

    joblib.dump(final_model, model_path)
    with open(classes_path, "wb") as f:
        pickle.dump(class_names, f)

    print(f"Saved final model to: {model_path}")
    print(f"Saved class labels to: {classes_path}")

    # 9. Write metrics report
    metrics_md = os.path.join(REPORTS_DIR, "model_b_metrics.md")
    with open(metrics_md, "w", encoding="utf-8") as f:
        f.write("# Model B -- IOGP Rule Tagging (prototype)\n\n")
        f.write(f"Training rows: {len(df_filtered)} "
                f"across {len(class_names)} classes.\n\n")
        f.write("| Class | Support |\n|---|---|\n")
        for rule, count in class_counts.items():
            f.write(f"| {rule} | {count} |\n")
        f.write(f"\n| Approach | Macro-F1 |\n|---|---|\n")
        f.write(f"| LogReg (multinomial) | {lr_mean:.3f} ± {lr_std:.3f} |\n")
        f.write(f"| XGBoost (multi) | {xgb_mean:.3f} ± {xgb_std:.3f} |\n")
        f.write(f"\nSelected: **{winner_name}**\n\n")
        f.write("Features: 768-dim DistilBERT embeddings only.\n")
        f.write("Classes merged per IOGP_MERGE in config.py.\n")
        f.write("Predictions below IOGP_MIN_CONFIDENCE are suppressed at "
                "inference and shown as 'Uncertain -- manual review required'.\n")
    print(f"Saved metrics to: {metrics_md}")

    print("\nDone.")


if __name__ == "__main__":
    main()