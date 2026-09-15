"""
src/diagnose_signal.py

Answers three questions with evidence, not opinion:

  TEST 1  Can the model trivially identify a row's SOURCE from its text?
          If AUC ~ 1.0, then source is a near-perfect proxy for the label
          and any cross-source metric is measuring style, not severity.

  TEST 2  Is there real severity signal INSIDE the IHM Stefanini data alone?
          (425 rows, 41% SIF, single source, single writing style.)
          This is the only clean like-for-like test available.

  TEST 3  Does adding OSHA + synthetic rows to training actually IMPROVE
          performance on held-out IHM rows, compared to TEST 2?
          This directly answers: "was the Path A data expansion worth it?"
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/labeled_data_expanded.csv")
X = np.load(ROOT / "data/processed/embeddings.npy")
y = df["is_sif"].to_numpy()
src = df["source"].to_numpy()
SEED = 42

print("=" * 78)
print("  SIGNAL DIAGNOSTIC")
print("=" * 78)
print(f"\nCorpus: {len(df):,} rows")
for s, c in pd.Series(src).value_counts().items():
    sub = y[src == s]
    print(f"  {s:<18} {c:>5} rows   {sub.mean()*100:5.1f}% SIF")


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                         subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
                         eval_metric="logloss", random_state=SEED,
                         n_jobs=-1, tree_method="hist")


def cv_auc(Xd, yd, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    aucs, aps, f2s = [], [], []
    for tr, te in skf.split(Xd, yd):
        m = xgb().fit(Xd[tr], yd[tr])
        p = m.predict_proba(Xd[te])[:, 1]
        aucs.append(roc_auc_score(yd[te], p))
        aps.append(average_precision_score(yd[te], p))
        f2s.append(fbeta_score(yd[te], (p >= 0.5).astype(int), beta=2,
                               zero_division=0))
    return np.mean(aucs), np.std(aucs), np.mean(aps), np.mean(f2s)


# ---------------- TEST 1 ----------------
print("\n" + "=" * 78)
print("  TEST 1  -- Is SOURCE trivially detectable from text?")
print("=" * 78)
is_osha = np.isin(src, ["OSHA_SIR", "OSHA_ABSTRACTS"]).astype(int)
a, s, _, _ = cv_auc(X, is_osha)
print(f"\n  AUC for predicting 'is this OSHA?'  :  {a:.3f} ± {s:.3f}")
is_syn = (src == "SYNTHETIC_NEG").astype(int)
a2, s2, _, _ = cv_auc(X, is_syn)
print(f"  AUC for predicting 'is synthetic?'  :  {a2:.3f} ± {s2:.3f}")
if a > 0.95 or a2 > 0.95:
    print("\n  [!] Source is near-perfectly separable from text alone.")
    print("      Because label ~ source in this corpus, cross-source metrics")
    print("      measure writing style, NOT hazard severity.")


# ---------------- TEST 2 ----------------
print("\n" + "=" * 78)
print("  TEST 2  -- Real severity signal inside IHM Stefanini alone")
print("=" * 78)
m_ihm = src == "IHM_Stefanini"
Xi, yi = X[m_ihm], y[m_ihm]
print(f"\n  {len(yi)} rows, {yi.mean()*100:.1f}% SIF  (single source/style)")
a, s, ap, f2 = cv_auc(Xi, yi)
print(f"\n  ROC-AUC : {a:.3f} ± {s:.3f}   (0.50 = random)")
print(f"  PR-AUC  : {ap:.3f}            (baseline = {yi.mean():.3f})")
print(f"  F2@0.5  : {f2:.3f}")
ihm_auc = a
if a >= 0.70:
    print("\n  [OK] Genuine severity signal exists in the real data.")
elif a >= 0.60:
    print("\n  [~] Weak but non-random signal.")
else:
    print("\n  [!] Essentially no signal from text alone at this data size.")


# ---------------- TEST 3 ----------------
print("\n" + "=" * 78)
print("  TEST 3  -- Does OSHA + synthetic augmentation HELP on IHM?")
print("=" * 78)
ihm_idx = np.where(m_ihm)[0]
other_idx = np.where(~m_ihm)[0]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

base_a, aug_a = [], []
for tr, te in skf.split(ihm_idx, y[ihm_idx]):
    tr_i, te_i = ihm_idx[tr], ihm_idx[te]
    m1 = xgb().fit(X[tr_i], y[tr_i])
    base_a.append(roc_auc_score(y[te_i], m1.predict_proba(X[te_i])[:, 1]))
    tr_aug = np.concatenate([tr_i, other_idx])
    m2 = xgb().fit(X[tr_aug], y[tr_aug])
    aug_a.append(roc_auc_score(y[te_i], m2.predict_proba(X[te_i])[:, 1]))

print(f"\n  Test set = held-out IHM rows (real labels, real text)\n")
print(f"  Train on IHM only            : ROC-AUC {np.mean(base_a):.3f} ± {np.std(base_a):.3f}")
print(f"  Train on IHM + OSHA + synth  : ROC-AUC {np.mean(aug_a):.3f} ± {np.std(aug_a):.3f}")
delta = np.mean(aug_a) - np.mean(base_a)
print(f"\n  Effect of augmentation       : {delta:+.3f} AUC")
if delta > 0.03:
    print("  [OK] Augmentation genuinely helps. Keep the expanded corpus.")
elif delta > -0.03:
    print("  [~] Augmentation is roughly neutral on real data.")
else:
    print("  [!] Augmentation HURTS. The extra data is adding noise, not signal.")

print("\n" + "=" * 78)
print("  VERDICT")
print("=" * 78)
print(f"  IHM-only signal (TEST 2)     : ROC-AUC {ihm_auc:.3f}")
print(f"  Augmentation effect (TEST 3) : {delta:+.3f}")
print("=" * 78)