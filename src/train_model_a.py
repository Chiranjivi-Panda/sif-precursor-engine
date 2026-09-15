"""
src/train_model_a.py -- Model A: SIF binary classifier (SINGLE-SOURCE, HONEST)

DATA DECISION
-------------
Trains on IHM Stefanini only (425 rows, 41% SIF, real potential-severity
labels on BOTH classes). The OSHA augmentation path was evaluated and
rejected: see reports/data_expansion_results.md.

  - Source was 99.9% detectable from text (label ~ source confound)
  - Source-held-out ROC-AUC collapsed to 0.497 (random)
  - Augmentation reduced real-data ROC-AUC by 0.061

GUARDS
------
  * Features = 768-dim DistilBERT embeddings ONLY (no metadata leakage)
  * Threshold tuned on an inner validation split, never on the test fold
  * Degeneracy guard rejects flag rates outside [0.15, 0.75]
  * ROC-AUC / PR-AUC reported so quality is visible independent of threshold
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "labeled_data.csv"
EMB = ROOT / "data" / "processed" / "embeddings.npy"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEED = 42
N_FOLDS = 5
MIN_FLAG, MAX_FLAG = 0.20, 0.65   # 41% prevalence; >65% flagged = no triage value
WEIGHTS = [0.75, 1.0, 1.5, 2.0]
THRESHOLDS = np.arange(0.25, 0.76, 0.025)


def metrics(y, pred):
    return dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=fbeta_score(y, pred, beta=1, zero_division=0),
        f2=fbeta_score(y, pred, beta=2, zero_division=0),
        flag_rate=float(np.mean(pred)),
    )


def summarise(name, runs):
    a = {k: np.array([r[k] for r in runs]) for k in runs[0]}
    print(f"| {name:<26} | {a['precision'].mean():.3f}±{a['precision'].std():.3f} "
          f"| {a['recall'].mean():.3f}±{a['recall'].std():.3f} "
          f"| {a['f1'].mean():.3f}±{a['f1'].std():.3f} "
          f"| {a['f2'].mean():.3f}±{a['f2'].std():.3f} "
          f"| {a['flag_rate'].mean():.2f} |")
    return {k: (v.mean(), v.std()) for k, v in a.items()}


def make_xgb(w):
    # Conservative capacity: 340 training rows cannot support a deep ensemble.
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.06,
        subsample=0.8, colsample_bytree=0.6,
        reg_lambda=3.0, reg_alpha=0.5, min_child_weight=3,
        scale_pos_weight=w, eval_metric="logloss",
        random_state=SEED, n_jobs=-1, tree_method="hist",
    )


def fit_tuned(X, y, verbose=False):
    """
    Tune scale_pos_weight and the decision threshold using OUT-OF-FOLD
    predictions from the SAME calibrated pipeline that is returned.

    BUG FIXED HERE (see reports/limitations.md):
      The previous version tuned the threshold on RAW XGBoost probabilities
      but returned a SIGMOID-CALIBRATED model. Platt scaling shifts the
      probability scale, so the chosen threshold did not transfer -- three of
      five folds ended up flagging ~100% of reports despite a degeneracy
      guard. Tuning on out-of-fold calibrated probabilities removes the
      mismatch entirely.
    """
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best = None

    for w in WEIGHTS:
        pipe = CalibratedClassifierCV(make_xgb(w), method="sigmoid", cv=3)
        # OOF probabilities on the SAME scale the final model will produce
        oof = cross_val_predict(pipe, X, y, cv=inner,
                                method="predict_proba", n_jobs=1)[:, 1]
        for t in THRESHOLDS:
            pred = (oof >= t).astype(int)
            fr = pred.mean()
            if not (MIN_FLAG <= fr <= MAX_FLAG):
                continue
            f2 = fbeta_score(y, pred, beta=2, zero_division=0)
            if best is None or f2 > best[0]:
                best = (f2, w, t)

    f2v, w, t = best or (0.0, 1.0, 0.5)
    if verbose:
        print(f"    tuned: weight={w}  threshold={t:.3f}  oof_F2={f2v:.3f}")

    model = CalibratedClassifierCV(make_xgb(w), method="sigmoid", cv=3)
    model.fit(X, y)
    return model, float(t)


def main():
    print("=" * 92)
    print("  MODEL A -- SIF BINARY CLASSIFIER   (IHM Stefanini only, honest)")
    print("=" * 92)

    df = pd.read_csv(DATA)
    X = np.load(EMB)
    assert len(df) == len(X), f"Row mismatch: {len(df)} vs {len(X)}"
    y = df["is_sif"].to_numpy()
    txt = df["Description"].to_numpy()

    print(f"\nCorpus : {len(df)} rows, {X.shape[1]}-dim embeddings")
    print(f"         {int(y.sum())} SIF ({y.mean()*100:.1f}%) / "
          f"{int((y==0).sum())} non-SIF")
    print(f"         Single source, real labels on both classes.")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    xg, tf_runs, aucs, aps = [], [], [], []
    example = None

    print("\nFolds:")
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        model, thr = fit_tuned(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        pred = (prob >= thr).astype(int)
        xg.append(metrics(y[te], pred))
        aucs.append(roc_auc_score(y[te], prob))
        aps.append(average_precision_score(y[te], prob))

        v = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                            stop_words="english", min_df=2)
        Ttr = v.fit_transform(txt[tr])
        lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                C=0.5, random_state=SEED).fit(Ttr, y[tr])
        tf_runs.append(metrics(y[te], lr.predict(v.transform(txt[te]))))

        print(f"  fold {fold}: thr={thr:.3f}  flag={pred.mean():.2f}  "
              f"AUC={aucs[-1]:.3f}")
        if fold == 0:
            example = (te, prob, pred, thr)

    hdr = (f"\n| {'Approach':<26} | {'Precision':<11} | {'Recall':<11} "
           f"| {'F1':<11} | {'F2':<11} | Flag |")
    bar = "-" * 92

    print("\n" + "=" * 92)
    print(f"  5-FOLD CV   ({y.mean()*100:.1f}% SIF -- the true class balance)")
    print(f"  Always-yes baseline: F2 = "
          f"{5*y.mean()/(4*y.mean()+1):.3f}, precision = {y.mean():.3f}")
    print("=" * 92 + hdr + "\n" + bar)
    summarise("TF-IDF + LogReg", tf_runs)
    res = summarise("DistilBERT + XGBoost", xg)
    print(bar)

    print("\n" + "=" * 92)
    print("  THRESHOLD-FREE DISCRIMINATION")
    print("=" * 92)
    print(f"  ROC-AUC : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  PR-AUC  : {np.mean(aps):.3f} ± {np.std(aps):.3f}   "
          f"(baseline = {y.mean():.3f})")

    te, prob, pred, thr = example
    print("\n" + "=" * 92)
    print(f"  EXAMPLES  (fold 0, threshold = {thr:.3f})")
    print("=" * 92)
    tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    for lab, c in [("TRUE POSITIVE", (y[te] == 1) & (pred == 1)),
                   ("TRUE NEGATIVE", (y[te] == 0) & (pred == 0)),
                   ("FALSE POSITIVE", (y[te] == 0) & (pred == 1)),
                   ("FALSE NEGATIVE", (y[te] == 1) & (pred == 0))]:
        w = np.where(c)[0]
        print(f"\n[{lab}]" + (f"  prob={prob[w[0]]:.3f}\n  {txt[te[w[0]]][:190]}..."
                             if len(w) else "  none"))

    print("\n" + "=" * 92)
    print("  FINAL PRODUCTION MODEL")
    print("=" * 92)
    final, ft = fit_tuned(X, y, verbose=True)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "model_a_sif_classifier.joblib")
    joblib.dump({"mode": "embeddings_only", "threshold": ft,
                 "embedding_dim": int(X.shape[1])},
                MODELS / "model_a_preprocessor.joblib")
    print(f"  Saved. Threshold = {ft:.3f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_a_metrics.md").write_text(
        "# Model A -- Honest Single-Source Metrics\n\n"
        f"Data: IHM Stefanini, {len(df)} rows, {y.mean()*100:.1f}% SIF.\n"
        f"Features: 768-dim DistilBERT embeddings only.\n"
        f"5-fold stratified CV. Threshold {ft:.3f} tuned on inner validation.\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Precision | {res['precision'][0]:.3f} ± {res['precision'][1]:.3f} |\n"
        f"| Recall | {res['recall'][0]:.3f} ± {res['recall'][1]:.3f} |\n"
        f"| F1 | {res['f1'][0]:.3f} ± {res['f1'][1]:.3f} |\n"
        f"| **F2** | **{res['f2'][0]:.3f} ± {res['f2'][1]:.3f}** |\n"
        f"| Flag rate | {res['flag_rate'][0]:.3f} |\n"
        f"| ROC-AUC | {np.mean(aucs):.3f} ± {np.std(aucs):.3f} |\n"
        f"| PR-AUC | {np.mean(aps):.3f} |\n\n"
        f"Always-yes baseline F2 = {5*y.mean()/(4*y.mean()+1):.3f}.\n",
        encoding="utf-8")
    print("  Saved -> reports/model_a_metrics.md\n")


if __name__ == "__main__":
    main()