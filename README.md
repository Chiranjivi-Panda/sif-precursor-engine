# SIF Precursor Detection Engine
**Smart India Hackathon 2026 — Problem Statement SIH26165**

An AI/NLP system that automatically detects Serious Injury & Fatality (SIF) potential 
in safety incident reports and tags them to IOGP Life-Saving Rules, enabling 
prioritized HSE interventions in oil & gas operations.

**Team:** BitManiacs — Rajiv Gandhi Institute of Petroleum Technology (RGIPT)

**Reference Architecture:** Based on Parikh et al. (2024), *Automatic identification 
of incidents involving potential serious injuries and fatalities (PSIF)*, 
Scientific Reports 14:8091.

## Section 2: Problem Statement

Oil India receives thousands of safety incident and near-miss reports. While non-fatal accidents have decreased by 51%, fatalities have only decreased by 25% because the causes of fatalities are fundamentally different from minor injuries. This engine automatically identifies reports that had the potential for a Serious Injury or Fatality (SIF), allowing HSE teams to intervene before a fatal accident occurs.

## Section 3: System Architecture

```text
Data → Embedding (DistilBERT CLS) → Model A (SIF Binary) → Model B (IOGP Multi-class) → Dashboard
```
- **Model A**: XGBoost binary classifier, F2-optimized for high recall
- **Model B**: Logistic Regression multi-class, trained on manually-mapped IOGP rules
- **Dashboard**: Streamlit, 4 pages (Live Predictor, Aggregate Dashboard, Batch Upload, Limitations)

## Section 4: Dataset

- Kaggle IHM Stefanini mining safety dataset (425 reports, 2016-2017)
- Used as proxy for OIL's oil & gas UA/UC data (not publicly available)
- 165 reports mapped to 6 IOGP Life-Saving Rule categories after filtering

## Section 5: Key Results

- **Model A (SIF Binary)**: TF-IDF+LogReg baseline F2=0.66±0.05, DistilBERT+XGBoost F2=0.56±0.08 (on 425-row dataset)
- **Model B (IOGP Multi-class)**: LogReg macro-F1=0.57±0.00 (outperformed XGBoost on small per-class counts)
- **Inference speed**: ~0.1-0.2s per report (CPU)

## Section 6: Installation & Usage

```bash
# Clone repository
git clone <your-repo-url>
cd sif-precursor-engine

# Install dependencies
pip install -r requirements.txt

# Place raw data
# (Download IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv 
#  from Kaggle and place in data/raw/ as safety_data_main.csv)

# Run data preparation
python src/run_data_prep.py

# Generate embeddings (takes ~2-5 min on first run)
python src/run_embed.py

# Train models
python src/train_model_a.py
python src/train_model_b.py
python src/precompute_scores.py  # Note: Must be re-run whenever either model is retrained

# Test inference
python src/test_predict.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Section 7: Repository Structure

```text
sif-precursor-engine/
├── data/
│   ├── raw/                  # Original CSV files
│   └── processed/            # Labeled data + cached embeddings
├── src/
│   ├── config.py             # IOGP mapping, constants, paths
│   ├── data_prep.py          # Data loading & target engineering
│   ├── embed.py              # DistilBERT CLS token extraction
│   ├── train_model_a.py      # SIF binary classifier training
│   ├── train_model_b.py      # IOGP multi-class training
│   ├── precompute_scores.py  # Cache dashboard Aggregate page scores
│   └── predict.py            # Unified inference function
├── models/                   # Trained .joblib/.pkl files
├── dashboard/
│   └── app.py                # Streamlit multi-page UI
├── reports/
│   ├── eda/                  # Exploratory analysis plots
│   └── limitations.md        # Known constraints & assumptions
└── README.md
```

## Section 8: Limitations & Future Work

Please see [`reports/limitations.md`](reports/limitations.md) for full details on known constraints.
- **Highlights**: Domain gap (mining → oil & gas), small dataset (425 rows), heuristic IOGP mapping.
- **Future work**: Retrain on OIL's real data, fine-tune embeddings on safety corpus, adaptive threshold tuning.

## Section 9: References

- Parikh, P., Penfield, J., & Juaire, M. (2024). DOI: 10.1038/s41598-024-58824-y
- IOGP Life-Saving Rules: https://www.iogp.org/life-savingrules/
- Kaggle dataset: https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database
