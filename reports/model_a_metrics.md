# Model A -- Honest Single-Source Metrics

Data: IHM Stefanini, 425 rows, 41.2% SIF.
Features: 768-dim DistilBERT embeddings only.
5-fold stratified CV. Threshold 0.375 tuned on inner validation.

| Metric | Value |
|---|---|
| Precision | 0.525 ± 0.019 |
| Recall | 0.714 ± 0.120 |
| F1 | 0.601 ± 0.047 |
| **F2** | **0.663 ± 0.084** |
| Flag rate | 0.560 |
| ROC-AUC | 0.688 ± 0.025 |
| PR-AUC | 0.612 |

Always-yes baseline F2 = 0.778.
