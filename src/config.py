"""
Configuration constants for the SIF Precursor Engine.

This module defines all shared constants: paths, model parameters,
label mappings, and the IOGP rule taxonomy.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────
# IOGP Life-Saving Rule mapping
#
# Maps each 'Critical Risk' value in the raw dataset to a standardised
# IOGP Life-Saving Rule category.
#
# Usage:  iogp_rule = IOGP_MAP.get(critical_risk_value, "Uncategorized")
# Anything not explicitly listed falls back to "Uncategorized".
# ──────────────────────────────────────────────────────────────────────
IOGP_MAP: dict[str, str] = {
    # Energy Isolation
    "Pressurized Systems":                          "Energy Isolation",
    "Pressurized Systems / Chemical Substances":    "Energy Isolation",
    "Power lock":                                   "Energy Isolation",
    "Blocking and isolation of energies":           "Energy Isolation",
    "Electrical Shock":                             "Energy Isolation",
    "Electrical installation":                      "Energy Isolation",
    # Confined space merged into Energy Isolation — only n=1 observation
    # in the raw dataset, too few to form its own IOGP category.
    "Confined space":                               "Energy Isolation",

    # Safe Mechanical Lifting
    "Suspended Loads":                              "Safe Mechanical Lifting",
    "Machine Protection":                           "Safe Mechanical Lifting",

    # Driving / Line of Fire
    "Vehicles and Mobile Equipment":                "Driving/Line of Fire",
    "Traffic":                                      "Driving/Line of Fire",

    # Working at Heights
    "Fall":                                         "Working at Heights",
    "Fall prevention":                              "Working at Heights",
    "Fall prevention (same level)":                 "Working at Heights",

    # Hazardous Materials
    "Chemical substances":                          "Hazardous Materials",
    "Liquid Metal":                                 "Hazardous Materials",
    "Burn":                                         "Hazardous Materials",
    "Poll":                                         "Hazardous Materials",

    # Line of Fire
    "Pressed":                                      "Line of Fire",
    "Cut":                                          "Line of Fire",
    "Manual Tools":                                 "Line of Fire",
    "Projection":                                   "Line of Fire",
    "Projection of fragments":                      "Line of Fire",
    "Projection/Burning":                           "Line of Fire",
    "Projection/Choco":                             "Line of Fire",
    "Projection/Manual Tools":                      "Line of Fire",
    "Plates":                                       "Line of Fire",
    "remains of choco":                             "Line of Fire",

    # Uncategorized — risks that don't map cleanly to an IOGP rule
    "Bees":                                         "Uncategorized",
    "Venomous Animals":                             "Uncategorized",
    "Individual protection equipment":              "Uncategorized",
    "Not applicable":                               "Uncategorized",
    "Others":                                       "Uncategorized",
}
# ──────────────────────────────────────────────────────────────────────
# Model B class consolidation
#
# The 6-class taxonomy leaves 'Safe Mechanical Lifting' (n=8) and
# 'Driving/Line of Fire' (n=9) with too few examples to learn -- the former
# scored F1=0.00 in CV while still being predicted at 0.74 confidence.
#
# We merge them into their nearest hazard family. Both involve uncontrolled
# movement of a mass toward a person, which is the Line of Fire concept.
# This is a documented prototype simplification; with OIL's production data
# the full 9-rule IOGP taxonomy becomes viable.
# ──────────────────────────────────────────────────────────────────────
IOGP_MERGE: dict[str, str] = {
    "Safe Mechanical Lifting": "Line of Fire",
    "Driving/Line of Fire":    "Line of Fire",
}

# Minimum confidence before a rule tag is shown to a user.
# Below this the UI must display "Uncertain -- manual review required".
IOGP_MIN_CONFIDENCE: float = 0.45

# ──────────────────────────────────────────────────────────────────────
# SIF severity threshold
#
# Potential Accident Level values that qualify as SIF (Serious Injury
# or Fatality).  Everything else is non-SIF.
# ──────────────────────────────────────────────────────────────────────
SIF_LEVELS: list[str] = ["IV", "V", "VI"]
# ──────────────────────────────────────────────────────────────────────
# Model B class consolidation
#
# The 6-class taxonomy leaves 'Safe Mechanical Lifting' (n=8) and
# 'Driving/Line of Fire' (n=9) with too few examples to learn. In 3-fold CV
# 'Safe Mechanical Lifting' scored F1 = 0.00 while still being predicted at
# 0.74 confidence on live input -- a confidently wrong tag, which is the worst
# failure mode for a triage tool.
#
# Both merged classes involve uncontrolled movement of a mass toward a person,
# which is the Line of Fire concept. This is a documented prototype
# simplification; OIL's production data will support the full 9-rule taxonomy.
# ──────────────────────────────────────────────────────────────────────
IOGP_MERGE: dict[str, str] = {
    "Safe Mechanical Lifting": "Line of Fire",
    "Driving/Line of Fire":    "Line of Fire",
}

# Minimum softmax confidence before a rule tag is shown to a user.
# Softmax always sums to 1.0, so a "confident" score can still come from a
# class the model never learned. Below this floor the UI shows
# "Uncertain -- manual review required" instead of a specific rule.
IOGP_MIN_CONFIDENCE: float = 0.55

# ──────────────────────────────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────────────────────────────
# Actual model used in src/embed.py and src/predict.py.
# 768-dim [CLS] token embedding, matching Parikh et al. (2024).
EMBEDDING_MODEL_NAME: str = "distilbert-base-uncased"

# ──────────────────────────────────────────────────────────────────────
# Paths (all relative to the project root)
#
# Raw data inventory (3 files in data/raw/):
#   1. safety_data_main.csv
#      ↳ Renamed from IHMStefanini_..._with_accidents_description.csv
#      ↳ ONLY file used for training — contains "Description" free-text.
#   2. safety_data_no_description_IGNORE.csv
#      ↳ Renamed from IHMStefanini_..._database.csv
#      ↳ NO Description column — NEVER load in data_prep or training.
#   3. osha_template_reference_ONLY.xlsx
#      ↳ Blank OSHA-301 form template — zero data rows.
#      ↳ NEVER load into a dataframe. Visual reference for dashboard only.
# ──────────────────────────────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_CSV_PATH       = os.path.join(PROJECT_ROOT, "data", "raw", "safety_data_main.csv")
# ──────────────────────────────────────────────────────────────────────
# Active training corpus
#
# Path A (OSHA merge) was built, evaluated, and REJECTED.
# See reports/data_expansion_results.md for the full write-up.
#
#   - 'is this row from OSHA?' was predictable from text at ROC-AUC 0.999,
#     making source a near-perfect proxy for the label.
#   - Source-held-out test (train OSHA+synthetic, test IHM): ROC-AUC 0.497,
#     i.e. random. Flagged 404 of 411 reports.
#   - Augmentation REDUCED real-data ROC-AUC: 0.650 -> 0.589 (-0.061).
#
# Reproduce with: python src/diagnose_signal.py
#
# REJECTED:
# PROCESSED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled_data_expanded.csv")

PROCESSED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled_data.csv")
EMBEDDINGS_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "embeddings.npy")
MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
