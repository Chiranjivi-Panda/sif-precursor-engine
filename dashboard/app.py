"""
SIF Precursor Detection Engine Dashboard

Streamlit web application for interactive inference and analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_report, load_artifacts, set_artifacts
from src.config import PROCESSED_CSV_PATH
import traceback

# -------------------------------------------------------------------
# Helper / Caching Functions
# -------------------------------------------------------------------

@st.cache_resource
def init_app():
    return load_artifacts()

def render_footer():
    st.markdown("---")
    st.caption("SIF-SENTINEL — Team BitManiacs, RGIPT | SIH 2026 (SIH26165)<br>"
               "Prototype trained on a public industrial-safety proxy dataset. Not production-calibrated for oil & gas operations.<br>"
               "[GitHub Repository](https://github.com/Chiranjivi-Panda/sif-precursor-engine)", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists(PROCESSED_CSV_PATH):
        return pd.read_csv(PROCESSED_CSV_PATH)
    return pd.DataFrame()

SCORED_REPORTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'scored_reports.csv')

@st.cache_data
def load_and_score_top_10():
    """Reads precomputed scored_reports.csv and returns the top 10 highest-risk."""
    if not os.path.exists(SCORED_REPORTS_PATH):
        return pd.DataFrame()
        
    df = pd.read_csv(SCORED_REPORTS_PATH)
    if df.empty:
        return pd.DataFrame()
        
    scored = []
    for idx, row in df.iterrows():
        desc = str(row.get('Description', ''))
        sif_prob = row.get('sif_probability', 0.0)
        iogp_rule = row.get('pred_iogp_rule', '')
        
        scored.append({
            'Report text': desc[:100] + ('...' if len(desc) > 100 else ''),
            'SIF probability (%)': round(sif_prob * 100, 2),
            'Predicted IOGP rule': iogp_rule if pd.notna(iogp_rule) and iogp_rule else "N/A",
            'Site (Local)': row.get('Local', 'Unknown')
        })
        
    scored_df = pd.DataFrame(scored)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(by='SIF probability (%)', ascending=False).head(10)
    return scored_df

# -------------------------------------------------------------------
# Page 1: Live Predictor
# -------------------------------------------------------------------
def page_live_predictor():
    st.header("Live Predictor")
    
    df = load_data()
    
    # Optional metadata dropdowns
    col1, col2 = st.columns(2)
    with col1:
        ind_sectors = ["None"] + sorted(df['Industry Sector'].dropna().unique().tolist()) if not df.empty else ["None", "Mining", "Metals", "Others"]
        sector_choice = st.selectbox("Industry Sector (Optional)", options=ind_sectors)
    with col2:
        emp_types = ["None"] + sorted(df['Employee or Third Party'].dropna().unique().tolist()) if not df.empty else ["None", "Employee", "Third Party", "Third Party (Remote)"]
        emp_choice = st.selectbox("Employee Type (Optional)", options=emp_types)
        
    text_input = st.text_area(
        "Enter safety incident report description",
        height=200,
        placeholder="Example: Worker was operating forklift when..."
    )
    
    if st.button("Analyze Report"):
        if not text_input.strip():
            st.warning("Please enter a report description")
            return
            
        with st.spinner("Analyzing report with DistilBERT + XGBoost..."):
            s = sector_choice if sector_choice != "None" else None
            e = emp_choice if emp_choice != "None" else None
            
            if len(text_input) > 5000:
                st.info("Input text exceeded 5000 characters and was truncated.")
                text_input = text_input[:5000]
                
            try:
                res = predict_report(text_input, industry_sector=s, employee_type=e)
                
                st.markdown("### Prediction Results")
                
                # A) SIF Flag Badge
                if res['is_sif']:
                    st.error("⚠️ **SIF POTENTIAL DETECTED**")
                else:
                    st.success("✓ **Non-SIF Report**")
                    
                st.write(f"**SIF probability:** {res['sif_probability'] * 100:.1f}%")
                
                # B) IOGP Rule Tag
                st.markdown("#### IOGP Life-Saving Rule")
                if res['is_sif'] and res['iogp_rule']:
                    st.warning(f"🏷️ **{res['iogp_rule']}** (Confidence: {res['iogp_confidence'] * 100:.1f}%)")
                else:
                    st.markdown("<span style='color:gray'>No IOGP tagging (non-SIF report)</span>", unsafe_allow_html=True)
                    
                # C) Similar Past Reports
                st.markdown("---")
                with st.expander("3 Most Similar Historical Reports", expanded=True):
                    for i, nn in enumerate(res['nearest_neighbors']):
                        st.markdown(f"**Neighbor {i+1}**")
                        short_text = nn['text'][:150] + ("..." if len(nn['text']) > 150 else "")
                        st.write(f"_{short_text}_")
                        
                        # Badge for actual label
                        sif_color = "red" if nn['is_sif'] == 1 else "green"
                        sif_text = "SIF" if nn['is_sif'] == 1 else "Non-SIF"
                        
                        st.markdown(
                            f"<span style='background-color:{sif_color};color:white;padding:2px 6px;border-radius:4px;font-size:12px;'>{sif_text}</span> "
                            f"**Actual IOGP:** {nn['iogp_rule']}", 
                            unsafe_allow_html=True
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
            except Exception as ex:
                st.error("Could not process this report. Please check the input text and try again.")
                traceback.print_exc()

    render_footer()


# -------------------------------------------------------------------
# Page 2: Aggregate Dashboard
# -------------------------------------------------------------------
def page_aggregate_dashboard():
    st.header("Aggregate Dashboard")
    df = load_data()
    
    if df.empty:
        st.error("Training data not found.")
        return
        
    # --- Site-Level SIF Density ---
    st.subheader("Site-Level SIF Density")
    site_stats = df.groupby('Local').agg(
        total=('is_sif', 'count'),
        sif_count=('is_sif', 'sum')
    ).reset_index()
    site_stats['sif_density'] = (site_stats['sif_count'] / site_stats['total']) * 100
    site_stats = site_stats.sort_values(by='sif_density', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(site_stats['Local'], site_stats['sif_density'], color='steelblue')
    
    # Annotate and hatch for n < 10
    for i, bar in enumerate(bars):
        n = site_stats.iloc[i]['total']
        if n < 10:
            bar.set_alpha(0.4)
            bar.set_hatch('//')
        
        # Annotation
        density = site_stats.iloc[i]['sif_density']
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{density:.0f}% (n={n})",
            ha='center', va='bottom', fontsize=9
        )
        
    ax.set_ylabel("% SIF Reports")
    ax.set_ylim(0, 115) # Give room for labels
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    st.caption("Sites with <10 reports shown in lighter color/hatch (statistically unreliable)")
    
    # --- IOGP Rule Distribution ---
    st.markdown("---")
    st.subheader("IOGP Rule Distribution")
    iogp_counts = df['iogp_rule'].value_counts().reset_index()
    iogp_counts.columns = ['iogp_rule', 'count']
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=iogp_counts, x='iogp_rule', y='count', ax=ax2, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    ax2.set_xlabel("")
    ax2.set_ylabel("Count of Reports")
    st.pyplot(fig2)
    st.caption("'Uncategorized' represents reports not mapped to standard IOGP Life-Saving Rules")
    
    # --- Industry Sector SIF Density ---
    st.markdown("---")
    st.subheader("Industry Sector SIF Density")
    ind_stats = df.groupby('Industry Sector').agg(
        total=('is_sif', 'count'),
        sif_count=('is_sif', 'sum')
    ).reset_index()
    ind_stats['sif_density'] = (ind_stats['sif_count'] / ind_stats['total']) * 100
    ind_stats = ind_stats.sort_values(by='sif_density', ascending=False)
    
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars3 = ax3.bar(ind_stats['Industry Sector'], ind_stats['sif_density'], color='coral')
    
    for i, bar in enumerate(bars3):
        n = ind_stats.iloc[i]['total']
        if n < 10:
            bar.set_alpha(0.4)
            bar.set_hatch('//')
        density = ind_stats.iloc[i]['sif_density']
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{density:.0f}% (n={n})",
            ha='center', va='bottom', fontsize=9
        )
    ax3.set_ylabel("% SIF Reports")
    ax3.set_ylim(0, 115)
    st.pyplot(fig3)
    
    # --- Priority List Table ---
    st.markdown("---")
    st.subheader("Top 10 High-Risk Reports (Predicted)")
    st.caption("These reports should be prioritized for HSE review")
    
    with st.spinner("Scoring reports..."):
        top_10_df = load_and_score_top_10()
        if top_10_df.empty:
            st.warning("Precomputed scores not found. Please run `python src/precompute_scores.py`.")
        else:
            st.dataframe(top_10_df, use_container_width=True, hide_index=True)
            st.caption("Note: These scores are in-sample predictions from the production model, which was trained on all 425 reports. They illustrate the ranking interface, not model performance. All reported metrics (Model A F2, Model B macro-F1) come from held-out cross-validation folds — see the Limitations page.")
    
    render_footer()


# -------------------------------------------------------------------
# Page 3: Batch Upload & Scoring
# -------------------------------------------------------------------
def page_batch_upload():
    st.header("Batch Upload & Scoring")
    
    uploaded_file = st.file_uploader("Upload CSV with 'Description' column", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
            
        if 'Description' not in df.columns and 'description' not in df.columns:
            # Check case-insensitive match
            desc_col = next((c for c in df.columns if c.lower() == 'description'), None)
            if not desc_col:
                st.error("Error: CSV must contain a 'Description' column with report text")
                return
        else:
            desc_col = 'Description' if 'Description' in df.columns else 'description'
            
        st.success(f"CSV loaded successfully with {len(df)} rows.")
        
        if st.button("Score All Rows"):
            if len(df) > 200:
                st.warning("Free-tier resource limit: only the first 200 rows were scored. The full pipeline has no such limit when deployed on OIL infrastructure.")
                df = df.head(200)

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total = len(df)
            
            for i, row in df.iterrows():
                # Update progress
                progress_bar.progress(int((i / total) * 100))
                status_text.text(f"Processing row {i+1} of {total}...")
                
                text = str(row[desc_col])
                if not text or text.lower() == 'nan':
                    results.append({'is_sif': False, 'sif_probability': 0.0, 'iogp_rule': None, 'iogp_confidence': None})
                    continue
                    
                # We optionally pass Industry Sector / Employee Type if they exist in the CSV
                ind = row.get('Industry Sector', None)
                emp = row.get('Employee or Third Party', None)
                
                try:
                    res = predict_report(text, industry_sector=ind, employee_type=emp)
                    results.append({
                        'is_sif': res['is_sif'],
                        'sif_probability': res['sif_probability'],
                        'iogp_rule': res['iogp_rule'],
                        'iogp_confidence': res['iogp_confidence']
                    })
                except Exception as ex:
                    traceback.print_exc()
                    st.error(f"Could not process row {i+1}. Please check the input text and try again.")
                    results.append({'is_sif': False, 'sif_probability': 0.0, 'iogp_rule': None, 'iogp_confidence': None})
                
            progress_bar.progress(100)
            status_text.text("Scoring complete!")
            
            res_df = pd.DataFrame(results)
            df_final = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            # Summary stats
            total_processed = len(df_final)
            flagged = df_final['is_sif'].sum()
            pct_flagged = (flagged / total_processed) * 100 if total_processed > 0 else 0
            
            st.markdown("### Summary Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reports", total_processed)
            col2.metric("Flagged as SIF", flagged)
            col3.metric("% Flagged", f"{pct_flagged:.1f}%")
            
            st.markdown("### Scored Results")
            st.dataframe(df_final, use_container_width=True)
            
            # Download button
            csv_export = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Scored CSV",
                data=csv_export,
                file_name="scored_reports.csv",
                mime="text/csv"
            )

    render_footer()


# -------------------------------------------------------------------
# Page 4: Limitations & Model Card
# -------------------------------------------------------------------
def page_limitations():
    st.header("Limitations & Model Card")
    
    st.markdown("### Dataset & Domain Gap")
    st.markdown("This prototype is trained on 425 mining/metals safety reports (Kaggle IHM Stefanini dataset) as a proxy for OIL India's oil & gas UA/UC data, which was not available pre-Finale. The model architecture is domain-agnostic and will be retrained on OIL's actual reports with no structural changes.")

    st.markdown("### Proxy Dataset Labeling Differences")
    st.markdown("The Kaggle mining dataset's severity labels may not perfectly align with OIL's SIF criteria. For example, a sulfuric acid facial burn was labeled non-SIF in the training data (likely based on actual outcome rather than worst-case potential). The nearest-neighbor feature helps human reviewers catch such cases. With OIL's actual data labeled by their HSE team, this mismatch will not occur.")
    
    st.markdown("### IOGP Mapping is Heuristic")
    st.markdown("The Life-Saving Rule tags were manually derived from the dataset's 'Critical Risk' field, as no such mapping exists in public data or the reference paper (Parikh et al. 2024). Model B's accuracy ceiling is bounded by this initial heuristic mapping quality.")
    
    st.markdown("### Class Sparsity in Model B")
    st.markdown("Several IOGP categories have small sample sizes (8-22 examples). Categories with <10 examples show unreliable metrics. With production-scale data, per-rule counts will support finer-grained tagging.")
    
    st.markdown("### Validation Approach")
    st.markdown("All metrics use stratified k-fold cross-validation (5-fold for Model A, 3-fold for Model B) and report mean ± std, not single train/test splits, given the small dataset size.")
    st.markdown("- **Dashboard Scores:** The 'Top 10 High-Risk Reports' shown on the Aggregate Dashboard are in-sample predictions for UI demonstration purposes. They do not represent held-out model performance.")
    
    st.markdown("### Classification Threshold")
    st.markdown("SIF threshold is fixed at 50% for this prototype. Adaptive threshold tuning is a planned enhancement, done carefully to avoid overfitting.")
    
    st.markdown("### Ethical Exclusions")
    st.markdown("Gender ('Genre' column) was deliberately excluded from both models as a predictive feature, as it has no causal relationship to incident severity and would raise fairness concerns.")
    
    st.info("For questions or technical details, see the full project documentation in `/reports/`.")
    render_footer()


def main():
    st.set_page_config(page_title="SIF Precursor Detection Engine", layout="wide")
    
    with st.spinner("Loading AI models — this takes ~30 seconds on first load..."):
        artifacts = init_app()
        set_artifacts(artifacts)
    
    page = st.sidebar.radio("Navigation", 
        ["Live Predictor", "Aggregate Dashboard", "Batch Upload", "Limitations"])

    if page == "Live Predictor":
        page_live_predictor()
    elif page == "Aggregate Dashboard":
        page_aggregate_dashboard()
    elif page == "Batch Upload":
        page_batch_upload()
    elif page == "Limitations":
        page_limitations()

if __name__ == "__main__":
    main()
