import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@st.cache_data
def load_metrics():
    with open(DATA_DIR / "model_metrics.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_anova():
    return pd.read_parquet(DATA_DIR / "selection_bias_anova.parquet")

@st.cache_data
def load_train_sample():
    return pd.read_parquet(DATA_DIR / "features_train_sample.parquet")

@st.cache_data
def load_category_cycle_summary():
    return pd.read_parquet(DATA_DIR / "category_cycle_summary.parquet")

@st.cache_data
def load_acquisition_insights():
    return pd.read_parquet(DATA_DIR / "acquisition_insights.parquet")

@st.cache_data
def load_retention_insights():
    return pd.read_parquet(DATA_DIR / "retention_insights.parquet")

@st.cache_data
def load_feature_importance_acq():
    return pd.read_parquet(DATA_DIR / "feature_importance_acq.parquet")

@st.cache_data
def load_feature_importance_ret():
    return pd.read_parquet(DATA_DIR / "feature_importance_ret.parquet")

@st.cache_data
def load_calibration_acq():
    return pd.read_parquet(DATA_DIR / "calibration_acq.parquet")

@st.cache_data
def load_calibration_ret():
    return pd.read_parquet(DATA_DIR / "calibration_ret.parquet")

def load_optimal_threshold_sim():
    # Cache completely removed to prevent stale JSON keys
    return pd.read_parquet(DATA_DIR / "optimal_threshold_sim.parquet")

def load_optimal_threshold_metrics():
    # Cache completely removed to prevent stale JSON keys
    with open(DATA_DIR / "optimal_threshold_metrics.json", "r") as f:
        return json.load(f)

def load_shap_values_acq():
    return pd.read_parquet(DATA_DIR / "shap_values_acq.parquet")

def load_shap_data_acq():
    return pd.read_parquet(DATA_DIR / "shap_data_acq.parquet")

def load_shap_values_ret():
    return pd.read_parquet(DATA_DIR / "shap_values_ret.parquet")

def load_shap_data_ret():
    return pd.read_parquet(DATA_DIR / "shap_data_ret.parquet")

