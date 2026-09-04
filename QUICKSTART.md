# Quick Start Guide

## Prerequisites
- Python 3.8+
- ~500 MB disk space (for models)

## 3-Minute Demo

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download data:
   - Go to https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database
   - Download the CSV
   - Place as `data/raw/safety_data_main.csv`

3. Run data prep + embedding (one-time, ~5 min):
   ```bash
   python src/data_prep.py
   python src/run_embed.py
   ```

4. Launch dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
   Open http://localhost:8501 in your browser

5. Try the Live Predictor with this test case:
   > "Worker fell 15 meters from scaffolding due to harness failure"
   
   Should flag as SIF + tag "Working at Heights"
