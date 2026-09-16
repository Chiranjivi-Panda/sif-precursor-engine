"""
SIF Precursor Detection Engine Dashboard

An enterprise-grade, highly optimized, and visually stunning web application 
for real-time SIF precursor triage, aggregate HSE insights, and governance auditing.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback

from src.predict import predict_report, load_artifacts, set_artifacts
from src.config import PROCESSED_CSV_PATH, IOGP_MIN_CONFIDENCE

# -------------------------------------------------------------------
# Helper / Caching Functions
# -------------------------------------------------------------------

@st.cache_resource
def init_app():
    return load_artifacts()

def render_footer():
    st.markdown("""
    <div class="footer-container">
        <hr style="border-color: rgba(255, 255, 255, 0.1); margin-top: 3rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 0; flex-wrap: wrap;">
            <div style="color: rgba(241, 242, 246, 0.5); font-size: 13px;">
                <strong>SIF-SENTINEL</strong> — Team BitManiacs, RGIPT | SIH 2026 (SIH26165)
            </div>
            <div style="color: rgba(241, 242, 246, 0.4); font-size: 12px; text-align: right;">
                Prototype trained on public industrial-safety proxy database. Optimized for OIL India deployment. <br>
                <a href="https://github.com/Chiranjivi-Panda/sif-precursor-engine" target="_blank" style="color: #1e90ff; text-decoration: none;">GitHub Repository</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            'Report Text Summary': desc[:100] + ('...' if len(desc) > 100 else ''),
            'SIF Probability (%)': round(sif_prob * 100, 1),
            'Predicted IOGP Rule': iogp_rule if pd.notna(iogp_rule) and iogp_rule else "Uncategorized",
            'Site Identifier': row.get('Local', 'Unknown')
        })
        
    scored_df = pd.DataFrame(scored)
    if not scored_df.empty:
        scored_df = scored_df.sort_values(by='SIF Probability (%)', ascending=False).head(10)
    return scored_df

# -------------------------------------------------------------------
# Custom CSS for Glassmorphism, Animations, and Industrial Theme
# -------------------------------------------------------------------
def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        /* Core layout and gradient background overrides */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background: linear-gradient(135deg, #070a1e 0%, #0c152b 50%, #060b18 100%) !important;
            color: #f1f2f6 !important;
        }
        
        /* Modernized Sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(6, 10, 26, 0.9) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(15px);
        }
        
        /* Global Glassmorphic Card Styling */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
            animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(30, 144, 255, 0.15);
            border: 1px solid rgba(30, 144, 255, 0.25);
        }
        
        /* Keyframe Animations */
        @keyframes fadeSlideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes pulse-sif {
            0% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.6); }
            70% { box-shadow: 0 0 0 15px rgba(255, 71, 87, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }
        }
        
        @keyframes subtle-blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        /* Badge Styling */
        .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            letter-spacing: 0.5px;
            padding: 10px 20px;
            border-radius: 30px;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }
        
        .badge-sif {
            background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: pulse-sif 2s infinite;
        }
        
        .badge-non-sif {
            background: linear-gradient(135deg, #2ed573 0%, #7bed9f 100%);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .badge-uncertain {
            background: linear-gradient(135deg, #ffa502 0%, #ff7f50 100%);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.15);
            animation: subtle-blink 2.5s infinite;
        }
        
        .badge-iogp {
            background: rgba(30, 144, 255, 0.15);
            color: #1e90ff;
            border: 1px solid rgba(30, 144, 255, 0.3);
            font-size: 12px;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: 600;
        }
        
        /* Interactive Form Elements */
        .tooltip-icon {
            display: inline-block;
            cursor: pointer;
            color: #1e90ff;
            margin-left: 6px;
            font-weight: bold;
        }
        
        /* Scenario Pills */
        .scenario-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #f1f2f6;
            border-radius: 10px;
            padding: 10px 15px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            width: 100%;
        }
        
        .scenario-btn:hover {
            background: rgba(30, 144, 255, 0.1);
            border-color: rgba(30, 144, 255, 0.4);
            transform: translateY(-2px);
        }
        
        /* Dynamic HTML Progress Bar Container */
        .custom-progress-container {
            width: 100%;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            height: 14px;
            margin: 10px 0 20px 0;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        
        .custom-progress-bar {
            height: 100%;
            border-radius: 10px;
            transition: width 1s cubic-bezier(0.1, 0.8, 0.1, 1);
        }

        /* Responsive Metric Card */
        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }
        .metric-card-val {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 2px;
        }
        .metric-card-lbl {
            color: rgba(241, 242, 246, 0.6);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
                /* Icon color fixes */
        svg {
            fill: rgba(241, 242, 246, 0.75) !important;
        }
        [data-testid="stFileUploaderDropzone"] svg,
        [data-testid="stExpander"] svg,
        [data-baseweb="radio"] svg {
            fill: #1e90ff !important;
        }
        [data-testid="stIconMaterial"] {
            color: #1e90ff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] div:first-child {
            border-color: #1e90ff !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State variables for interactive demo scenarios
if 'demo_text' not in st.session_state:
    st.session_state.demo_text = ""
if 'sector_val' not in st.session_state:
    st.session_state.sector_val = "None"
if 'emp_val' not in st.session_state:
    st.session_state.emp_val = "None"

def load_scenario(text, sector, emp):
    st.session_state.demo_text = text
    st.session_state.sector_val = sector
    st.session_state.emp_val = emp

# -------------------------------------------------------------------
# Page 1: Live Incident Triage
# -------------------------------------------------------------------
def render_hero_banner():
    st.markdown("""
    <div style="
        background: linear-gradient(180deg, rgba(7,10,30,0.55) 0%, rgba(6,11,24,0.97) 100%),
                    url('https://images.unsplash.com/photo-1581093588401-fbb62a02f120?w=1920&q=80') center center / cover no-repeat;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 3.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    ">
        <div style="font-size:13px; font-weight:700; color:#1e90ff; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">
            AI-Powered SIF Precursor Detection
        </div>
        <div style="font-size:40px; font-weight:700; line-height:1.15; color:#f1f2f6; margin-bottom:15px; letter-spacing:-1px;">
            See What Matters<br>Before It Matters Most
        </div>
        <div style="font-size:16px; color:rgba(241,242,246,0.7); max-width:700px; line-height:1.6;">
            Reduce risk, protect your people, and improve site performance through a connected, 
            high-throughput Environmental Health & Safety intelligence engine powered by human-centered AI.
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_live_predictor():
    st.markdown('<h1 style="font-weight:700; font-size:2.5rem; letter-spacing:-0.5px;"><span style="-webkit-text-fill-color: initial;">🔴</span> <span style="background:linear-gradient(135deg, #f1f2f6 0%, #a4b0be 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Live Incident Triage</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(241,242,246,0.65); font-size:1.1rem; margin-top:-10px;">Immediate AI assessment and hazard prioritization engine at submission time.</p>', unsafe_allow_html=True)
    render_hero_banner()   # <-- ADD THIS LINE
    df = load_data()
    
    # 3A. Demo Scenario Buttons
    st.markdown("### ⚡ Quick Demo Scenarios")
    
    scenarios = {
        "crusher": {
            "title": "⚙️ Crusher Jam",
            "text": "The technician was conducting repairs on the crushing machine's feed hopper. The power lock was bypassed. The machine cycled briefly, and the operator's hand was pinched in the mechanism, risking severe amputation.",
            "sector": "Mining",
            "emp": "Employee"
        },
        "heights": {
            "title": "🏗️ Heights Near-Miss",
            "text": "A scaffold builder was installing high-altitude piping at approximately 15 meters on Platform D. While turning to grab an anchor bolt, the builder slipped. The safety harness lanyard anchor failed, leading to a near-fall caught only by luck.",
            "sector": "Metals",
            "emp": "Third Party"
        },
        "loto": {
            "title": "⚡ Panel Flashover",
            "text": "A subcontractor went into the live MCC room without a permit to work. During electrical inspection of the circuit panel, the worker dropped an uninsulated metal wrench onto the live terminal board, creating a heavy arc flash and minor shock.",
            "sector": "Others",
            "emp": "Third Party (Remote)"
        },
        "benign": {
            "title": "✅ Minor Clean Trip",
            "text": "Housekeeping reported a small amount of rain water had leaked through the roof panels onto the main administration floor lobby near the drinking water station. Danger signs placed, floors wiped clean.",
            "sector": "Mining",
            "emp": "Employee"
        }
    }
    
    sc_cols = st.columns(4)
    for idx, (key, info) in enumerate(scenarios.items()):
        with sc_cols[idx]:
            if st.button(info["title"], key=f"btn_{key}", use_container_width=True):
                load_scenario(info["text"], info["sector"], info["emp"])
                st.rerun()

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    # 3B. Input Form Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter Safety Incident Narrative Description",
            value=st.session_state.demo_text,
            height=220,
            placeholder="Type a detailed safety report here (e.g., 'A pressure valve blew during routine gas pipeline inspection...')"
        )
    
    with col2:
        st.markdown('**Operational Context** <span title="Collected for operational context only. Decoupled from predictive model features to prevent statistical bias.">ℹ️</span>', unsafe_allow_html=True)
        
        # Pull distinct sectors safely
        ind_sectors = ["None"] + sorted(df['Industry Sector'].dropna().unique().tolist()) if not df.empty else ["None", "Mining", "Metals", "Others"]
        try:
            sector_idx = ind_sectors.index(st.session_state.sector_val)
        except ValueError:
            sector_idx = 0
            
        sector_choice = st.selectbox("Industry Sector", options=ind_sectors, index=sector_idx)
        
        emp_types = ["None"] + sorted(df['Employee or Third Party'].dropna().unique().tolist()) if not df.empty else ["None", "Employee", "Third Party", "Third Party (Remote)"]
        try:
            emp_idx = emp_types.index(st.session_state.emp_val)
        except ValueError:
            emp_idx = 0
            
        emp_choice = st.selectbox("Employee Role Category", options=emp_types, index=emp_idx)
        
        st.markdown('<p style="color:rgba(241,242,246,0.4); font-size:11px; margin-top:5px;">💡 Dropdown selections do not affect neural semantic representations. Hazard metrics are calculated purely from text mechanics.</p>', unsafe_allow_html=True)

    if st.button("🚀 Analyze Narrative Integrity", use_container_width=True):
        if not text_input.strip():
            st.warning("Please input or select an incident scenario to begin.")
            return
            
        with st.spinner("Processing narrative tokens through DistilBERT embedding layers & calibrated XGBoost pipeline..."):
            s = sector_choice if sector_choice != "None" else None
            e = emp_choice if emp_choice != "None" else None
            
            if len(text_input) > 5000:
                text_input = text_input[:5000]
                
            try:
                # Backend inference hook
                res = predict_report(text_input, industry_sector=s, employee_type=e)
                
                # Dynamic Results Area (Glassmorphic)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                st.markdown("### 📊 Model Prediction Outcomes")
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    # SIF Status Badge with pulse
                    if res['is_sif']:
                        st.markdown('<div class="badge badge-sif">⚠️ SIF PRECURSOR FOUND</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="badge badge-non-sif">✓ LOW RISK / NON-SIF</div>', unsafe_allow_html=True)
                
                with res_col2:
                    # Dynamic CSS Progress Bar + Probability Label
                    prob = res['sif_probability']
                    prob_pct = prob * 100
                    bar_color = "linear-gradient(90deg, #ff4757, #ff6b81)" if prob >= 0.5 else "linear-gradient(90deg, #2ed573, #7bed9f)"
                    
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 15px; font-weight:600; color:rgba(241,242,246,0.85);">SIF Potential Probability:</span>
                            <span style="font-size: 20px; font-weight:700; color:{'#ff4757' if prob >= 0.5 else '#2ed573'};">{prob_pct:.1f}%</span>
                        </div>
                        <div class="custom-progress-container">
                            <div class="custom-progress-bar" style="width: {prob_pct}%; background: {bar_color};"></div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # IOGP Life-Saving Rule Section
                st.markdown("---")
                st.markdown("#### 🏷️ Mapped IOGP Life-Saving Rule")
                
                if res['is_sif'] and res['iogp_rule']:
                    rule = res['iogp_rule']
                    conf = res['iogp_confidence']
                    conf_pct = conf * 100
                    
                    if conf >= IOGP_MIN_CONFIDENCE:
                        st.markdown(f"""
                            <div style="display: flex; align-items: center; gap: 15px; background: rgba(30,144,255,0.06); padding: 12px 20px; border-radius: 12px; border: 1px dashed rgba(30,144,255,0.3);">
                                <span class="badge-iogp" style="font-size:14px; padding: 6px 15px;">{rule}</span>
                                <span style="font-size:14px; color:rgba(241,242,246,0.8);">Neural Confidence Score: <strong style="color:#1e90ff;">{conf_pct:.1f}%</strong></span>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="badge badge-uncertain">⚠ UNCERTAIN CATEGORY</div>
                            <div style="font-size: 13.5px; color: rgba(241, 242, 246, 0.65); margin-top:-5px; padding-left: 5px;">
                                Neural confidence fell below safety threshold (<strong style="color:#ffa502;">55%</strong>). System forced manual HSE supervisor classification to prevent erroneous hazard mapping.
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:rgba(241,242,246,0.45); font-style: italic;'>No IOGP assignment triggered (Report scored as Non-SIF / low severity level)</span>", unsafe_allow_html=True)
                
                # 3D. Nearest Neighbors
                st.markdown("---")
                st.markdown("#### 🔍 Analogous Historical Incidents — For Sanity Check")
                st.markdown('<p style="color:rgba(241,242,246,0.5); font-size:12.5px; margin-top:-10px; margin-bottom:15px;">Leverages vector space cosine similarity to fetch historical references and establish contextual transparency.</p>', unsafe_allow_html=True)
                
                nn_cols = st.columns(3)
                for i, nn in enumerate(res['nearest_neighbors']):
                    with nn_cols[i]:
                        # Handle text truncation
                        short_text = nn['text'][:120] + ("..." if len(nn['text']) > 120 else "")
                        
                        sif_label = "SIF Potential" if nn['is_sif'] == 1 else "Non-SIF"
                        sif_bg = "rgba(255, 71, 87, 0.15)" if nn['is_sif'] == 1 else "rgba(46, 213, 115, 0.15)"
                        sif_txt_color = "#ff4757" if nn['is_sif'] == 1 else "#2ed573"
                        sif_border = "1px solid rgba(255, 71, 87, 0.3)" if nn['is_sif'] == 1 else "1px solid rgba(46, 213, 115, 0.3)"
                        
                        st.markdown(f"""
                            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 15px; height: 100%; min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
                                <div>
                                    <div style="font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px; color:rgba(241,242,246,0.4); font-weight:700; margin-bottom:8px;">Historical Analogy #{i+1}</div>
                                    <div style="font-size: 13px; font-style: italic; color: rgba(241,242,246,0.85); line-height:1.4; margin-bottom:12px;">"{short_text}"</div>
                                </div>
                                <div>
                                    <div style="display: inline-block; background: {sif_bg}; border: {sif_border}; color: {sif_txt_color}; font-size: 10.5px; padding: 3px 8px; border-radius: 6px; font-weight: 700; text-transform: uppercase; margin-bottom:6px;">{sif_label}</div>
                                    <div style="font-size:12px; color:rgba(241,242,246,0.5);"><strong>Rule:</strong> {nn['iogp_rule'] if nn['iogp_rule'] else 'Uncategorized'}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True) # Close glass card
                
            except Exception as ex:
                st.error("Exception triggered during vector inference loop.")
                with st.expander("Diagnostic Trace"):
                    st.code(traceback.format_exc())

    render_footer()

# -------------------------------------------------------------------
# Page 2: Executive HSSE Analytics
# -------------------------------------------------------------------
def page_aggregate_dashboard():
    st.markdown('<h1 style="font-weight:700; font-size:2.5rem; letter-spacing:-0.5px;"><span style="-webkit-text-fill-color: initial;">📊</span> <span style="background:linear-gradient(135deg, #f1f2f6 0%, #a4b0be 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Executive HSSE Analytics</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(241,242,246,0.65); font-size:1.1rem; margin-top:-10px;">Aggregates local site-level exposure density and surfaces hazardous operational patterns.</p>', unsafe_allow_html=True)
    
    df = load_data()
    if df.empty:
        st.error("Reference training dataset not found. Please verify directories.")
        return
        
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # 4A. Site-by-Site SIF Density Chart (Hero)
    st.subheader("📍 Site-Level SIF Precursor Density")
    st.markdown('<p style="color:rgba(241,242,246,0.5); font-size:13px; margin-top:-10px;">Interactive site-by-site hazard exposure analysis. Sites with small samples (&lt;10 total reports) are visually de-emphasized to prevent statistical reporting noise.</p>', unsafe_allow_html=True)
    
    site_stats = df.groupby('Local').agg(
        total=('is_sif', 'count'),
        sif_count=('is_sif', 'sum')
    ).reset_index()
    site_stats['sif_density'] = (site_stats['sif_count'] / site_stats['total']) * 100
    site_stats = site_stats.sort_values(by='sif_density', ascending=True)
    
    # Color mapping for Plotly
    # Low n: Muted transparent grey
    # High n, high risk: Vibrant Crimson/Red gradient
    # High n, low risk: Emerald/Blue
    colors = []
    text_labels = []
    for idx, row in site_stats.iterrows():
        n = int(row['total'])
        density = float(row['sif_density'])
        text_labels.append(f"{density:.1f}%<br>(n={n})")
        if n < 10:
            colors.append('rgba(164, 176, 190, 0.25)') # Low sample grey
        else:
            if density > 50:
                colors.append('rgba(255, 71, 87, 0.85)') # High Risk Red
            else:
                colors.append('rgba(30, 144, 255, 0.75)') # Baseline Blue
                
    fig_site = go.Figure()
    fig_site.add_trace(go.Bar(
        y=site_stats['Local'],
        x=site_stats['sif_density'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.1)', width=1)
        ),
        text=text_labels,
        textposition='outside',
        textfont=dict(color='#f1f2f6', size=11),
        hoverinfo='x+y+text'
    ))
    
    fig_site.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="% SIF Precursor Exposure",
            color='#a4b0be',
            gridcolor='rgba(255,255,255,0.05)',
            range=[0, 115]
        ),
        yaxis=dict(
            title="Operational Site ID",
            color='#a4b0be',
            gridcolor='rgba(255,255,255,0.05)'
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        showlegend=False
    )
    st.plotly_chart(fig_site, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Donut Chart Grid
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("🛡️ Mapped IOGP Life-Saving Rules")
        
        iogp_counts = df['iogp_rule'].value_counts().reset_index()
        iogp_counts.columns = ['iogp_rule', 'count']
        
        fig_donut = px.pie(
            iogp_counts,
            names='iogp_rule',
            values='count',
            hole=0.55,
            color_discrete_sequence=['#ff4757', '#1e90ff', '#2ed573', '#ffa502', '#5352ed', '#747d8c']
        )
        
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#f1f2f6')),
            margin=dict(l=10, r=10, t=10, b=10),
            height=300
        )
        fig_donut.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='rgba(0,0,0,0.3)', width=1))
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_g2:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("📊 Key Exposure Statistics")
        st.markdown('<p style="color:rgba(241,242,246,0.5); font-size:12px; margin-top:-10px;">Operational metrics calculated on current proxy repository context.</p>', unsafe_allow_html=True)
        
        total_reps = len(df)
        total_sif = df['is_sif'].sum()
        sif_ratio = (total_sif / total_reps) * 100 if total_reps > 0 else 0
        uncat_count = (df['iogp_rule'] == "Uncategorized").sum()
        uncat_ratio = (uncat_count / total_reps) * 100 if total_reps > 0 else 0
        
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top:10px;">
                <div class="metric-card">
                    <div class="metric-card-val" style="color:#1e90ff;">{total_reps}</div>
                    <div class="metric-card-lbl">Total Reports</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-val" style="color:#ff4757;">{total_sif}</div>
                    <div class="metric-card-lbl">SIF Precursors</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-val" style="color:#ff6b81;">{sif_ratio:.1f}%</div>
                    <div class="metric-card-lbl">Overall SIF Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-val" style="color:#ffa502;">{uncat_count}</div>
                    <div class="metric-card-lbl">Uncategorized Rules</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4D. Priority Queue Table
    st.markdown('<div class="glass-card" style="margin-top:1.5rem;">', unsafe_allow_html=True)
    st.subheader("🚨 Top 10 High-Risk Inspection Priority Queue")
    st.markdown('<p style="color:rgba(241,242,246,0.5); font-size:13px; margin-top:-10px;">Real-time queue identifying current highest risk exposures. Highly recommended for immediate, prioritized safety supervisor site audits.</p>', unsafe_allow_html=True)
    
    with st.spinner("Scoring database priority index..."):
        top_10_df = load_and_score_top_10()
        if top_10_df.empty:
            st.warning("Precomputed scored records path not found. Please execute the `src/precompute_scores.py` initialization pipeline script.")
        else:
            # Render using modern Streamlit styled display matching CSS palette
            st.dataframe(
                top_10_df.style.background_gradient(
                    subset=['SIF Probability (%)'],
                    cmap='Reds',
                    vmin=0,
                    vmax=100
                ),
                use_container_width=True,
                hide_index=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    render_footer()

# -------------------------------------------------------------------
# Page 3: Bulk Batch Ingestion
# -------------------------------------------------------------------
def page_batch_upload():
    st.markdown('<h1 style="font-weight:700; font-size:2.5rem; letter-spacing:-0.5px;"><span style="-webkit-text-fill-color: initial;">📁</span> <span style="background:linear-gradient(135deg, #f1f2f6 0%, #a4b0be 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Bulk Batch Ingestion</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(241,242,246,0.65); font-size:1.1rem; margin-top:-10px;">Ingest, evaluate, and prioritize entire backlogs of historical safety reports in seconds.</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Drop raw historical CSV logs below:", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV structure: {e}")
            return
            
        # Dynamically search for narrative columns
        desc_col = None
        for col in df.columns:
            if col.lower() in ['description', 'narrative', 'report_text', 'report text', 'incident_description']:
                desc_col = col
                break
                
        if not desc_col:
            st.warning("Automatic narrative column detection failed.")
            desc_col = st.selectbox("Select the narrative column to score:", options=df.columns)
        else:
            st.success(f"Successfully identified narrative column: **'{desc_col}'**")
            
        st.markdown(f"**Loaded Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns.")
        
        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        
        if st.button("🏁 Initiate Parallel Batch Score Pipeline", use_container_width=True):
            # Limit processing size on free resources to protect compute
            limit_cap = 200
            is_capped = False
            orig_len = len(df)
            if orig_len > limit_cap:
                df = df.head(limit_cap)
                is_capped = True
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total = len(df)
            
            t0 = time.time()
            
            # Loop-based processing for safe visual update
            for i, row in df.iterrows():
                progress_bar.progress(int((i / total) * 100))
                status_text.text(f"Scoring record {i+1} of {total}...")
                
                text = str(row[desc_col])
                if not text or text.lower() == 'nan' or len(text.strip()) == 0:
                    results.append({
                        'is_sif': False, 
                        'sif_probability': 0.0, 
                        'iogp_rule': 'Uncategorized', 
                        'iogp_confidence': 0.0
                    })
                    continue
                
                ind = row.get('Industry Sector', None)
                emp = row.get('Employee or Third Party', None)
                
                try:
                    res = predict_report(text, industry_sector=ind, employee_type=emp)
                    results.append({
                        'is_sif': res['is_sif'],
                        'sif_probability': res['sif_probability'],
                        'iogp_rule': res['iogp_rule'] if res['iogp_rule'] else 'Uncategorized',
                        'iogp_confidence': res['iogp_confidence'] if res['iogp_confidence'] else 0.0
                    })
                except Exception as ex:
                    results.append({
                        'is_sif': False, 
                        'sif_probability': 0.0, 
                        'iogp_rule': 'Uncategorized', 
                        'iogp_confidence': 0.0
                    })
            
            t1 = time.time()
            dt = t1 - t0
            rec_per_sec = total / dt if dt > 0 else 0
            
            progress_bar.progress(100)
            status_text.empty()
            
            res_df = pd.DataFrame(results)
            df_final = pd.concat([df.reset_index(drop=True), res_df], axis=1)
            
            # Summary Metrics Grid (Styled)
            flagged = df_final['is_sif'].sum()
            pct_flagged = (flagged / len(df_final)) * 100
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📈 Performance & Pipeline Statistics")
            
            st.markdown(f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-top:10px;">
                    <div class="metric-card">
                        <div class="metric-card-val" style="color:#2ed573;">{len(df_final)}</div>
                        <div class="metric-card-lbl">Scored Records</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val" style="color:#ff4757;">{flagged}</div>
                        <div class="metric-card-lbl">Precursors Flagged</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val" style="color:#ff6b81;">{pct_flagged:.1f}%</div>
                        <div class="metric-card-lbl">Exposure Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-val" style="color:#1e90ff;">{rec_per_sec:.1f}/s</div>
                        <div class="metric-card-lbl">Throughput Speed</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if is_capped:
                st.warning(f"⚠️ Computational limit: Evaluated the first {limit_cap} rows of {orig_len} total. Full local capacity supports unthrottled ingestion.")
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Results table preview
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📋 Enriched Dataset Output Preview")
            st.dataframe(df_final.head(10), use_container_width=True)
            
            # Download trigger button
            csv_export = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored CSV Log Database",
                data=csv_export,
                file_name="sif_triage_scored_logs.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

    render_footer()

# -------------------------------------------------------------------
# Page 4: Achievements & Roadmap (Reworked)
# -------------------------------------------------------------------
def page_achievements():
    st.markdown('<h1 style="font-weight:700; font-size:2.5rem; letter-spacing:-0.5px;"><span style="-webkit-text-fill-color: initial;">🏆</span> <span style="background:linear-gradient(135deg, #f1f2f6 0%, #a4b0be 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Achievements & Safety Roadmap</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(241,242,246,0.65); font-size:1.1rem; margin-top:-10px;">Engineering transparency, scientific rigor, and OIL India scale integration roadmap.</p>', unsafe_allow_html=True)
    
    # Grid Layout for key accomplishments
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💡 Key Achievements & Safety Innovations")
    st.markdown("""
    - **Pioneering Custom IOGP Life-Saving Rules Multi-Class Triage Layer:** Constructed and deployed an automated rule mapping pipeline for classification on raw, heterogeneous free text. Prior industrial research (including Parikh et al. 2024) left this unmapped.
    - **Eradication of Data Leakage Shortcuts:** Deployed deep invariant model structures by discarding context metadata (`Industry Sector`, `Employee Type`) from the neural input vector, preventing catastrophic model cheating and preserving safety-first semantic priority.
    - **Calibrated Out-Of-Fold Softmax Decision Gating:** Integrated confidence gating ($C_{\\text{floor}} = 0.55$) that filters out weak, overconfident classifications, redirecting ambiguous reports to human safety specialists.
    - **Strict Ethical Exclusions:** Eradicated demographic metadata (`Genre` / gender) from predictive layers to eliminate any potential statistical bias.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Benchmark Table Comparison
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Architectural Performance Evolution")
    st.markdown('<p style="color:rgba(241,242,246,0.5); font-size:13px; margin-top:-10px;">Quantifying improvements from prior baselines to our final validated model pipeline.</p>', unsafe_allow_html=True)
    
    metrics_data = {
        "Metric & Safeguard Controls": [
            "SIF Binary F₂ Score", 
            "SIF Discrimination (ROC-AUC)", 
            "Metadata Invariance (Leak Proofing)", 
            "Decision Threshold", 
            "Model B Class Layout", 
            "Incorrect IOGP Tag Handling"
        ],
        "Baseline Model (Mid-Session)": [
            "0.56 (Overfit / Single Split)", 
            "Not Evaluated / Unreported", 
            "Leaked (Metadata forced 99% ➔ 31% flip)", 
            "Hardcoded 0.5 Threshold", 
            "6 Classes (Severe Sparsity, F1=0.0)", 
            "Unsupervised (Quietly displayed wrong rules)"
        ],
        "Our Validated Model (Final Version)": [
            "0.663 ± 0.084 (5-Fold Stratified CV)", 
            "0.688 ± 0.025 (Strong, Honest Signal)", 
            "100% Secure (768-Dim Text Embeddings Only)", 
            "OOF-Tuned on Calibrated Outputs", 
            "4 Class Consolidated Taxonomy", 
            "Gated at 55% Confidence (Safe Fallbacks)"
        ],
        "Engineering Achievement / Significance": [
            "🔬 +18.4% gain with honest validation", 
            "🎯 True discriminative reliability", 
            "🛡️ Decision relies strictly on hazard narrative", 
            "⚙️ Prevents system degeneracy & triage fatigue", 
            "⚡ Stabilized minority class recall", 
            "🛑 Eliminates overconfident AI hallucinations"
        ]
    }
    
    st.table(pd.DataFrame(metrics_data))
    st.markdown('</div>', unsafe_allow_html=True)

    # OSHA Expansion Experiment Tab
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔬 Validation Rigor: The OSHA Data Expansion Experiment")
    st.markdown("""
    To pressure-test our design, we scaled our corpus from **425 to 2,211 reports** by integrating the **official US Government OSHA Severe Injury Reports database** (105,925 cleaned records) and **OSHA Accident Abstracts**. 
    
    Our initial models returned an outstanding **$F_2$ score of 0.92**, which most teams would immediately celebrate. However, we conducted rigorous diagnostics and exposed that:
    1. **Style Artifact Leaks:** The model reached a near-perfect ROC-AUC of **0.999** simply identifying *'Is this text from the OSHA database structure?'* rather than learning actual hazard severity.
    2. **Domain Collapse:** When evaluated on held-out safety logs, ROC-AUC collapsed to **0.497 (equivalent to random choice)**, flagging 98% of reports as false alarms.
    
    **Our Decision:** We prioritized safety integrity over deceptive metrics, **rejected the inflated $F_2=0.92$ model**, and reverted to the honest, clean proxy database. This scientific rigor guarantees that our pipeline design is structurally sound and ready to receive raw industrial data without catastrophic failure.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Future Roadmap Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🗺️ Future Scope & Production Integration with OIL India")
    st.markdown("""
    The transition from our validated prototype to OIL India's live platform utilizes a highly scalable integration roadmap:
    1. **Drop-In Data Swap:** The pipeline relies entirely on modular text embeddings; swapping our proxy dataset with OIL's historical UA/UC logs requires zero structural code re-engineering.
    2. **Language Adaptation (Domain BERT):** At production volume, we will apply domain-adaptive pre-training on DistilBERT utilizing OIL's specific vocabulary (*'wellhead'*, *'blowout preventer'*, *'hydrocarbon'*, *'derrick'*, *'sucker rod'*).
    3. **Taxonomy Expansion:** Massive production sample volume will allow expanding our consolidated 4-class set to the **full 9-rule standard IOGP taxonomy**, fully resolving class sparsity.
    4. **Barrier Failure Mining:** Deploing unsupervised Latent Dirichlet Allocation (LDA) and safety concept keyword mapping to parse unstructured text, enabling automated grouping of failure modes across operational assets.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    render_footer()

# -------------------------------------------------------------------
# Main Routing Logic
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="SIF Precursor Detection Engine", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject Custom CSS Styling Globally
    apply_custom_styles()
    
    # Load neural models & classification estimators once
    with st.spinner("Initializing Deep Neural Networks — This takes ~30 seconds on initial startup..."):
        artifacts = init_app()
        set_artifacts(artifacts)
    
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
            <div style="font-size: 24px; font-weight: 700; color: #1e90ff; letter-spacing: -1px;">SIF-SENTINEL</div>
            <div style="font-size: 11px; color: rgba(241,242,246,0.5); text-transform: uppercase; letter-spacing: 1px; margin-top:2px;">Precursor Detection Engine</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.08); margin-top: 0; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "NAVIGATION CONTROL", 
        [
            "🔴 Live Incident Triage", 
            "📊 HSSE Analytics", 
            "📁 Bulk Batch Ingestion", 
            "🏆 Achievements & Roadmap"
        ]
    )

    if page == "🔴 Live Incident Triage":
        page_live_predictor()
    elif page == "📊 HSSE Analytics":
        page_aggregate_dashboard()
    elif page == "📁 Bulk Batch Ingestion":
        page_batch_upload()
    elif page == "🏆 Achievements & Roadmap":
        page_achievements()

if __name__ == "__main__":
    main()