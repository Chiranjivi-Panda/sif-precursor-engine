# Limitations & Model Card

### Dataset & Domain Gap
This prototype is trained on 425 mining/metals safety reports (Kaggle IHM Stefanini dataset) as a proxy for OIL India's oil & gas UA/UC data, which was not available pre-Finale. The model architecture is domain-agnostic and will be retrained on OIL's actual reports with no structural changes.

### Proxy Dataset Labeling Differences
The Kaggle mining dataset's severity labels may not perfectly align with OIL's SIF criteria. For example, a sulfuric acid facial burn was labeled non-SIF in the training data (likely based on actual outcome rather than worst-case potential). The nearest-neighbor feature helps human reviewers catch such cases. With OIL's actual data labeled by their HSE team, this mismatch will not occur.

### IOGP Mapping is Heuristic
The Life-Saving Rule tags were manually derived from the dataset's 'Critical Risk' field, as no such mapping exists in public data or the reference paper (Parikh et al. 2024). Model B's accuracy ceiling is bounded by this initial heuristic mapping quality.

### Class Sparsity in Model B
Several IOGP categories have small sample sizes (8-22 examples). Categories with <10 examples show unreliable metrics. With production-scale data, per-rule counts will support finer-grained tagging.

### Validation Approach
All metrics use stratified k-fold cross-validation (5-fold for Model A, 3-fold for Model B) and report mean ± std, not single train/test splits, given the small dataset size.

### Classification Threshold
SIF threshold is fixed at 50% for this prototype. Adaptive threshold tuning is a planned enhancement, done carefully to avoid overfitting.

### Ethical Exclusions
Gender ('Genre' column) was deliberately excluded from both models as a predictive feature, as it has no causal relationship to incident severity and would raise fairness concerns.
