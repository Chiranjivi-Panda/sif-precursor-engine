# SIF Precursor Detection Engine — Executive Summary

## The Problem
Oil India receives thousands of safety reports monthly. Most describe minor incidents. 
A critical 20-25% describe situations that could have killed someone but luckily didn't. 
These get buried in manual review queues and spotted too late.

## Our Solution
An AI system that reads every report the moment it's filed and:
1. Flags it as SIF-potential (dangerous) or non-SIF (minor)
2. Tags it to one of 6 Life-Saving Rule categories (e.g., "Working at Heights", "Energy Isolation")
3. Shows a dashboard ranking sites/activities by SIF-precursor density

HSE teams can prioritize inspections where fatal risks are clustering.

## How It Works
- Based on a 2024 research paper (Parikh et al., VelocityEHS) that solved the same problem
- Uses DistilBERT (a natural language AI) to understand report text
- Uses XGBoost (a machine learning algorithm) to predict SIF potential
- Optimized for recall (better to over-flag than miss a real danger)

## Current Status
- Prototype trained on 425 mining safety reports (public Kaggle dataset)
- Achieves 66% F2 score (baseline) on SIF classification
- Dashboard provides live predictions + batch CSV scoring
- Inference speed: ~0.1-0.2 seconds per report

## Next Steps (If Advanced to Finale)
- Retrain on OIL's actual oil & gas UA/UC data (thousands of reports)
- Fine-tune embeddings on safety-specific language corpus
- Deploy on OIL's HSSE platform for real-time monitoring

## Key Strengths
- Transparent about limitations (small proxy dataset, domain gap)
- Reproducible architecture (exact replication of published research)
- Honest reporting (shows when baseline outperforms complex model)
- Built-in human review features (nearest-neighbor explanations)
