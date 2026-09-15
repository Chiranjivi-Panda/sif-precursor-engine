# Model B -- IOGP Rule Tagging (prototype)

Training rows: 165 across 4 classes.

| Class | Support |
|---|---|
| Line of Fire | 101 |
| Working at Heights | 22 |
| Hazardous Materials | 22 |
| Energy Isolation | 20 |

| Approach | Macro-F1 |
|---|---|
| LogReg (multinomial) | 0.463 ± 0.052 |
| XGBoost (multi) | 0.379 ± 0.059 |

Selected: **LogReg (multinomial)**

Features: 768-dim DistilBERT embeddings only.
Classes merged per IOGP_MERGE in config.py.
Predictions below IOGP_MIN_CONFIDENCE are suppressed at inference and shown as 'Uncertain -- manual review required'.
