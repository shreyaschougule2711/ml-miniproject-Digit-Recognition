"""
DigitAI — Handwritten Digit Recognition Studio
Interactive Web Application built with Streamlit.

A Supervised Learning System Comparing k-Nearest Neighbors, Linear SVM, and RBF Kernel SVM on MNIST.
Includes an End-to-End Multi-Digit Number Segmentation Pipeline.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from src.preprocessing import preprocess_canvas_image
from src.predict import predict_digit, are_models_available, load_model
from src.multi_digit import predict_multi_digit_number, segment_digits, preprocess_number_image
from src.data_loader import download_or_load_raw_mnist
from src.visualization import (
    plot_confusion_matrix,
    plot_average_digit_intensity,
    plot_metrics_bar_chart,
    plot_runtime_comparison,
)

# Optional import of drawable canvas
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


def render_st_canvas(**kwargs):
    """
    Renders st_canvas with adaptive keyword handling to support both legacy
    (<=0.9.3) and modern (>=0.10.0) versions of streamlit-drawable-canvas.
    Modern versions require return_image_data=True to serialize canvas pixels.
    """
    if not CANVAS_AVAILABLE:
        return None
    try:
        return st_canvas(**kwargs, return_image_data=True)
    except TypeError:
        return st_canvas(**kwargs)


def safe_get_canvas_image(canvas_obj) -> Optional[np.ndarray]:
    """
    Safely extracts image_data from a CanvasResult object, handling
    uninitialized states, type differences, and version-specific RuntimeErrors.
    """
    if canvas_obj is None or isinstance(canvas_obj, type):
        return None
    try:
        img_data = getattr(canvas_obj, "image_data", None)
        if img_data is not None and isinstance(img_data, np.ndarray) and img_data.size > 0:
            return img_data
    except Exception:
        return None
    return None


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="DigitAI — Handwritten Digit Recognition Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# APPLE-INSPIRED DESIGN SYSTEM (SCOPED CSS)
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --color-bg-base: #f8fafc;
        --color-card: #ffffff;
        --color-text-main: #0f172a;
        --color-text-sub: #475569;
        --color-text-muted: #64748b;
        --color-border: #e2e8f0;
        --color-border-subtle: #f1f5f9;
        --color-primary: #2563eb;
        --color-primary-hover: #1d4ed8;
        --color-primary-soft: #eff6ff;
        --color-primary-border: #bfdbfe;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --shadow-card: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        --shadow-card-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    }

    /* Enforce crisp light theme and contrast across Streamlit containers */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: var(--font-sans) !important;
    }

    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* Enforce high-contrast dark text on ALL markdown and HTML headings */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #0f172a !important;
        font-family: var(--font-sans) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Streamlit Main View Container padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }

    /* Top Brand / Page Titles */
    .page-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--color-primary-soft) !important;
        color: var(--color-primary) !important;
        border: 1px solid var(--color-primary-border) !important;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: #0f172a !important;
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #475569 !important;
        line-height: 1.5;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }

    /* Clean Card System */
    .apple-card {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-card);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .apple-card:hover {
        box-shadow: var(--shadow-card-hover);
    }

    .apple-card-compact {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
    }

    /* Metric Cards */
    .metric-card {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: var(--shadow-card);
        transition: transform 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--color-primary);
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Visual Workflow Steps */
    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 12px;
        margin: 1.2rem 0;
    }

    .workflow-step {
        background: #f8fafc;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.1rem;
        position: relative;
    }

    .workflow-num {
        display: inline-block;
        background: #e2e8f0;
        color: #334155;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }

    .workflow-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a !important;
        margin-bottom: 0.25rem;
    }

    .workflow-desc {
        font-size: 0.82rem;
        color: #475569 !important;
        line-height: 1.4;
    }

    /* Model Cards */
    .model-card {
        background: #ffffff !important;
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        height: 100%;
        box-shadow: var(--shadow-card);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .model-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }

    .model-tag {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        background: #f1f5f9;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .model-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a !important;
        margin-bottom: 0.4rem;
    }

    .model-card-desc {
        font-size: 0.86rem;
        color: #475569 !important;
        line-height: 1.45;
        margin-bottom: 0.8rem;
    }

    .model-spec-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        padding: 4px 0;
        border-top: 1px solid #f1f5f9;
        color: #475569;
    }

    /* Prominent Badges for Predictions */
    .digit-badge-hero {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #2563eb;
        color: #ffffff;
        font-size: 4.2rem;
        font-weight: 700;
        width: 108px;
        height: 108px;
        border-radius: 20px;
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.35);
        margin: 0.5rem 0;
    }

    .number-badge-hero {
        display: inline-block;
        background: #0f172a;
        color: #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        padding: 0.6rem 2.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        border: 2px solid #334155;
        margin: 0.6rem 0;
    }

    .status-pill-ok {
        background: #dcfce7;
        color: #15803d;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    .status-pill-missing {
        background: #fee2e2;
        color: #b91c1c;
        padding: 0.2rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Alternative Candidates Grid */
    .alt-candidate-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }

    /* Streamlit Tab and Button Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 0.92rem;
        color: #64748b;
    }

    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #2563eb !important;
        background: transparent !important;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        font-size: 0.92rem;
        transition: all 0.15s ease;
    }

    /* Sidebar refinement */
    section[data-testid="stSidebar"] {
        background-color: #fafbfc;
        border-right: 1px solid #eef2f6;
    }

    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a !important;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-brand-badge {
        font-size: 0.68rem;
        font-weight: 700;
        background: #2563eb;
        color: #ffffff;
        padding: 2px 6px;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }

    .sidebar-caption {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 2px;
        margin-bottom: 1rem;
    }

    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Table refinement */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .styled-table th {
        text-align: left;
        padding: 10px 12px;
        background: #f8fafc;
        border-bottom: 1px solid #e2e8f0;
        color: #475569;
        font-weight: 600;
    }
    .styled-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# DATA CACHING & METRICS LOADERS
# ==============================================================================
@st.cache_data
def get_cached_metadata() -> Optional[Dict[str, Any]]:
    """Loads training metadata if available."""
    if config.TRAINING_METADATA_PATH.exists():
        try:
            with open(config.TRAINING_METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data
def get_cached_metrics() -> Optional[Dict[str, Any]]:
    """Loads evaluation metrics if available."""
    if config.METRICS_JSON_PATH.exists():
        try:
            with open(config.METRICS_JSON_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_data
def get_cached_comparison_table() -> Optional[pd.DataFrame]:
    """Loads CSV comparison table if available."""
    if config.COMPARISON_CSV_PATH.exists():
        try:
            return pd.read_csv(config.COMPARISON_CSV_PATH)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner="Loading MNIST benchmark dataset (70,000 samples)...")
def get_cached_raw_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Loads or downloads raw MNIST dataset with Streamlit caching."""
    return download_or_load_raw_mnist()


# ==============================================================================
# DIAGNOSTIC PIPELINE INSPECTOR (DEBUG MODE)
# ==============================================================================
def render_debug_panel(res: dict, image_input, debug_mode: bool):
    """Renders the full diagnostic pipeline panel when debug_mode is enabled."""
    if not debug_mode or not res.get("success"):
        return

    st.markdown("---")
    st.markdown("### 🔧 Debug: Preprocessing Pipeline Inspector")

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        if isinstance(image_input, Image.Image):
            st.image(image_input, caption="Original Input Image", use_column_width=True)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                st.image(image_input, caption="Original Canvas (RGBA)", use_column_width=True)
            else:
                st.image(image_input, caption="Original Input (Grayscale)", use_column_width=True)
    with dcol2:
        gray = res.get("grayscale_image")
        if gray is not None:
            st.image(gray, caption="Grayscale (Polarity Corrected)", use_column_width=True)
    with dcol3:
        if "binary_image" in res:
            st.image(res["binary_image"], caption="Binary Threshold (Otsu)", use_column_width=True)

    if "annotated_image" in res:
        st.image(res["annotated_image"], caption="Detected Bounding Boxes (Left-to-Right)", use_column_width=True)

    digit_items = res.get("digits", [])
    if digit_items:
        st.markdown("#### Per-Digit 28×28 Normalized Crops")
        cols = st.columns(min(len(digit_items), 8))
        for idx, d in enumerate(digit_items):
            with cols[idx % len(cols)]:
                crop = d["crop_28x28"]
                img_scaled = Image.fromarray((crop * 255).astype(np.uint8)).resize((112, 112), Image.Resampling.NEAREST)
                st.image(img_scaled, caption=f"Digit #{d['index']}", width=112)
                st.caption(
                    f"Pred: **{d['predicted_digit']}**  \n"
                    f"Conf: {d['confidence_percentage']:.1f}%  \n"
                    f"Shape: {crop.shape}  \n"
                    f"Mean: {crop.mean():.4f}"
                )


# ==============================================================================
# SIDEBAR APPLICATION SHELL & NAVIGATION
# ==============================================================================
nav_options = [
    "Dashboard",
    "Single Digit Recognition",
    "Multi-Digit Recognition",
    "Dataset Explorer",
    "About System",
]

def navigate_to(page_name: str):
    """Callback for navigation buttons. Executes before widgets are instantiated."""
    st.session_state["current_page"] = page_name
    st.session_state["nav_radio_select"] = page_name

# Synchronize session state navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

if "nav_radio_select" not in st.session_state:
    st.session_state["nav_radio_select"] = st.session_state["current_page"]
elif st.session_state["nav_radio_select"] != st.session_state["current_page"]:
    st.session_state["nav_radio_select"] = st.session_state["current_page"]

def on_nav_change():
    st.session_state["current_page"] = st.session_state["nav_radio_select"]

# Sidebar Header Branding
st.sidebar.markdown(
    """
    <div style="padding-top: 0.5rem; margin-bottom: 0.2rem;">
        <div class="sidebar-brand-title">
            <span>✍️ DigitAI</span>
            <span class="sidebar-brand-badge">ENSEMBLE</span>
        </div>
        <div class="sidebar-caption">Ensemble Digit & Number Recognition</div>
    </div>
    """,
    unsafe_allow_html=True,
)

current_index = nav_options.index(st.session_state["current_page"]) if st.session_state["current_page"] in nav_options else 0

st.sidebar.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
nav_choice = st.sidebar.radio(
    "Select View",
    nav_options,
    index=current_index,
    key="nav_radio_select",
    on_change=on_nav_change,
    label_visibility="collapsed",
)
st.session_state["current_page"] = nav_choice

# Model Artifact Readiness in Sidebar
st.sidebar.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)
avail_status = are_models_available()
ens_ready = avail_status.get("ensemble", False)
ens_badge = '<span class="status-pill-ok">● Ready</span>' if ens_ready else '<span class="status-pill-missing">○ Missing</span>'
st.sidebar.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 0.84rem;">
        <span style="color: #334155; font-weight: 600;">Ensemble System</span>
        <span>{ens_badge}</span>
    </div>
    <div style="font-size: 0.74rem; color: #64748b; line-height: 1.35; margin-top: 2px;">
        Hard Majority Core: k-NN + Linear SVM + RBF SVM
    </div>
    """,
    unsafe_allow_html=True,
)

if not ens_ready:
    st.sidebar.info("💡 Train models via: `python src/train.py`")

st.sidebar.markdown('<div class="sidebar-section-title">Developer Tools</div>', unsafe_allow_html=True)
debug_mode = st.sidebar.checkbox(
    "Enable Diagnostic Mode",
    value=False,
    help="Displays intermediate binarization, contour masks, and per-digit crops.",
)

st.sidebar.markdown(
    """
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eef2f6; font-size: 0.76rem; color: #94a3b8;">
        DigitAI Studio • Yann LeCun MNIST Benchmark<br>
        Built with Streamlit & Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if st.session_state["current_page"] == "Dashboard":
    st.markdown('<div class="page-badge">✨ Unified Machine Learning Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Ensemble-Based Digit & Number Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A unified character recognition system combining k-Nearest Neighbors, Linear SVM, and RBF Kernel SVM via democratic hard majority voting on MNIST.</div>',
        unsafe_allow_html=True,
    )

    # Primary Action Launchpad Row
    col_act1, col_act2 = st.columns([1, 1], gap="medium")
    with col_act1:
        st.markdown(
            """
            <div class="apple-card" style="margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.4rem;">✍️</span>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 600; color: #0f172a;">Single Digit Recognition</h3>
                </div>
                <p style="font-size: 0.88rem; color: #64748b; line-height: 1.45; margin-bottom: 1rem;">
                    Draw isolated digits on an interactive canvas or upload handwritten images to test unified ensemble prediction in real time.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "👉 Open Single Digit Recognition",
            type="primary",
            use_container_width=True,
            key="btn_dash_single",
            on_click=navigate_to,
            args=("Single Digit Recognition",),
        )

    with col_act2:
        st.markdown(
            """
            <div class="apple-card" style="margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.4rem;">🔢</span>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 600; color: #0f172a;">Multi-Digit Recognition</h3>
                </div>
                <p style="font-size: 0.88rem; color: #64748b; line-height: 1.45; margin-bottom: 1rem;">
                    Draw or upload entire multi-digit numbers (e.g. 42, 2026, 987654) with automatic segmentation and sequential ensemble classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "👉 Open Multi-Digit Recognition",
            type="primary",
            use_container_width=True,
            key="btn_dash_multi",
            on_click=navigate_to,
            args=("Multi-Digit Recognition",),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Quantitative Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">70,000</div>
                <div class="metric-label">Benchmark Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">10</div>
                <div class="metric-label">Digit Classes (0–9)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">784</div>
                <div class="metric-label">Input Features (28×28 Grid)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        meta = get_cached_metadata()
        ens_acc = "96.05%" if not meta else f"{meta.get('results_summary', {}).get('ensemble', {}).get('accuracy', 0.9605)*100:.2f}%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{ens_acc}</div>
                <div class="metric-label">Ensemble Test Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # System Architecture Workflow Steps
    st.markdown("### 🔄 End-to-End System Workflow")
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-step">
                <span class="workflow-num">Step 1</span>
                <div class="workflow-title">Input Acquisition</div>
                <div class="workflow-desc">Capture stroke coordinates via responsive canvas or uploaded image stream.</div>
            </div>
            <div class="workflow-step">
                <span class="workflow-num">Step 2</span>
                <div class="workflow-title">Vision Preprocessing</div>
                <div class="workflow-desc">Invert polarity, crop bounding box, and scale to 20×20 preserving aspect ratio.</div>
            </div>
            <div class="workflow-step">
                <span class="workflow-num">Step 3</span>
                <div class="workflow-title">Center-of-Mass Alignment</div>
                <div class="workflow-desc">Translate image mass to center (14, 14) on 28×28 grid (LeCun standard).</div>
            </div>
            <div class="workflow-step">
                <span class="workflow-num">Step 4</span>
                <div class="workflow-title">Ensemble Majority Voting</div>
                <div class="workflow-desc">Democratic consensus across k-NN, Linear SVM, and RBF SVM with deterministic tie-breaking.</div>
            </div>
            <div class="workflow-step">
                <span class="workflow-num">Step 5</span>
                <div class="workflow-title">Unified Decision</div>
                <div class="workflow-desc">Output single predicted digit or sequence with calibrated confidence metrics.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Machine Learning Classifiers Profile
    st.markdown("### 🧠 Evaluated Machine Learning Paradigms")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        st.markdown(
            """
            <div class="model-card">
                <div>
                    <div class="model-card-header">
                        <span class="model-tag">Instance-Based</span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #16a34a;">Lazy Fit</span>
                    </div>
                    <div class="model-card-title">k-Nearest Neighbors</div>
                    <div class="model-card-desc">
                        Classifies query inputs by finding the <i>k</i> closest training instances in 784-D Euclidean space and performing distance-weighted majority voting.
                    </div>
                </div>
                <div>
                    <div class="model-spec-row"><span>Training Complexity</span><strong>O(1)</strong></div>
                    <div class="model-spec-row"><span>Inference Latency</span><strong>32.8 ms</strong></div>
                    <div class="model-spec-row"><span>Test Accuracy</span><strong>94.60%</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol2:
        st.markdown(
            """
            <div class="model-card">
                <div>
                    <div class="model-card-header">
                        <span class="model-tag">Linear Margin</span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #2563eb;">Ultra-Fast</span>
                    </div>
                    <div class="model-card-title">Linear SVM</div>
                    <div class="model-card-desc">
                        Constructs optimal maximum-margin hyperplanes separating digit classes with L2 regularization via LIBLINEAR coordinate descent.
                    </div>
                </div>
                <div>
                    <div class="model-spec-row"><span>Training Complexity</span><strong>Fast (LIBLINEAR)</strong></div>
                    <div class="model-spec-row"><span>Inference Latency</span><strong>10.8 ms</strong></div>
                    <div class="model-spec-row"><span>Test Accuracy</span><strong>90.55%</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol3:
        st.markdown(
            """
            <div class="model-card">
                <div>
                    <div class="model-card-header">
                        <span class="model-tag" style="background: #eff6ff; color: #1d4ed8;">Nonlinear Kernel</span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #1d4ed8;">Peak Accuracy</span>
                    </div>
                    <div class="model-card-title">RBF Kernel SVM</div>
                    <div class="model-card-desc">
                        Projects 784-D pixel vectors into an infinite-dimensional Hilbert space using Gaussian RBF kernels to capture complex curved strokes.
                    </div>
                </div>
                <div>
                    <div class="model-spec-row"><span>Training Complexity</span><strong>O(N² ~ N³)</strong></div>
                    <div class="model-spec-row"><span>Inference Latency</span><strong>4.0 ms</strong></div>
                    <div class="model-spec-row"><span>Test Accuracy</span><strong>96.55%</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mcol4:
        st.markdown(
            """
            <div class="model-card">
                <div>
                    <div class="model-card-header">
                        <span class="model-tag" style="background: #fef3c7; color: #b45309;">Ensemble Meta</span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: #d97706;">Consensus</span>
                    </div>
                    <div class="model-card-title">Ensemble Voting</div>
                    <div class="model-card-desc">
                        Combines k-NN, Linear SVM, and RBF SVM using hard majority voting with deterministic tie-breaking for cross-model consensus.
                    </div>
                </div>
                <div>
                    <div class="model-spec-row"><span>Voting Rule</span><strong>Hard Majority (2/3)</strong></div>
                    <div class="model-spec-row"><span>Base Models</span><strong>3 Classifiers</strong></div>
                    <div class="model-spec-row"><span>Test Accuracy</span><strong>96.05%</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: RECOGNIZE DIGITS (SINGLE DIGIT STUDIO)
# ==============================================================================
elif st.session_state["current_page"] == "Single Digit Recognition":
    st.markdown('<div class="page-badge">✍️ Single-Digit Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Single Digit Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Draw an isolated digit (0–9) or upload a handwriting sample for unified ensemble recognition.</div>',
        unsafe_allow_html=True,
    )

    if not all(avail_status.values()):
        missing = [k for k, v in avail_status.items() if not v]
        st.warning(
            f"⚠️ Model artifacts ({', '.join(missing)}) are missing on disk. Run `python src/train.py` or click below to train."
        )
        if st.button("🚀 Train Models Now (Stratified CPU Subset)", type="primary"):
            with st.spinner("Training models (k-NN, Linear SVM, RBF SVM)... Please wait ~60-90 seconds..."):
                from src.train import run_training_pipeline
                run_training_pipeline(train_samples=5000, test_samples=1500, tune=False)
                st.success("Models trained and serialized successfully!")
                st.cache_data.clear()
                st.rerun()

    # Primary Ensemble Badge (Single Unified Classifier)
    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 1.25rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.76rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">Primary Recognition Engine</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">Unified Ensemble Voting System</div>
            </div>
            <span class="status-pill-ok" style="font-size: 0.84rem; padding: 4px 12px;">Hard Majority Consensus (k-NN + Linear SVM + RBF SVM)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_key = "ensemble"

    input_tab1, input_tab2 = st.tabs(["✏️ Draw Digit on Canvas", "📁 Upload Digit Image"])

    active_image_input = None
    trigger_prediction = False

    # TAB 1: DRAW CANVAS
    with input_tab1:
        col_canvas_box, col_canvas_ctrl = st.columns([1, 1], gap="large")

        with col_canvas_ctrl:
            st.markdown("#### Canvas Settings")
            canvas_mode = st.radio(
                "Pen Color & Background",
                ["White on Black Background (MNIST Standard)", "Black on White Background"],
                horizontal=False,
                key="single_pen_color",
            )

            if "White on Black" in canvas_mode:
                stroke_color = "#FFFFFF"
                bg_color = "#000000"
            else:
                stroke_color = "#000000"
                bg_color = "#FFFFFF"

            stroke_width = st.slider("Stroke Thickness", min_value=12, max_value=32, value=22, step=2, key="single_stroke_width")
            st.caption("Tip: Draw large and clear in the center for optimal aspect-ratio scaling.")

        with col_canvas_box:
            st.markdown("#### Drawing Canvas")
            if CANVAS_AVAILABLE:
                canvas_result = render_st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="digit_canvas_single",
                )
            else:
                st.error("streamlit-drawable-canvas is not installed.")
                canvas_result = None

            if st.button("🔍 Recognize Drawn Digit", type="primary", use_container_width=True, key="btn_rec_draw"):
                img_data = safe_get_canvas_image(canvas_result)
                if img_data is not None:
                    active_image_input = img_data
                    trigger_prediction = True
                else:
                    st.warning("⚠️ Canvas is empty. Please draw a digit.")

    # TAB 2: UPLOAD IMAGE
    with input_tab2:
        st.markdown("#### Upload a Handwritten Digit Image")
        st.caption("Upload a single digit image (PNG, JPG, JPEG). The system will automatically invert polarity if needed and apply LeCun centering.")
        up_file = st.file_uploader("Choose single-digit image:", type=["png", "jpg", "jpeg"], key="single_up_file")

        if up_file is not None:
            up_img = Image.open(up_file)
            col_up_prev, col_up_action = st.columns([1, 2])
            with col_up_prev:
                st.image(up_img, caption="Uploaded Image", width=200)
            with col_up_action:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Recognize Uploaded Digit", type="primary", key="btn_rec_upload"):
                    active_image_input = up_img
                    trigger_prediction = True

    # RESULTS DISPLAY SECTION
    if trigger_prediction and active_image_input is not None:
        processed_28x28, flat_784, err_msg = preprocess_canvas_image(active_image_input)

        if err_msg:
            st.warning(f"⚠️ {err_msg}")
        else:
            with st.spinner("Classifying with trained model..."):
                pred_res = predict_digit(selected_key, flat_784)

            if not pred_res.get("success"):
                st.error(pred_res.get("error", "Prediction failed."))
            else:
                pred_digit = pred_res["predicted_digit"]
                conf_pct = pred_res["confidence_percentage"]
                score_type = pred_res["score_type"]
                explanation = pred_res["explanation"]

                st.markdown("---")
                st.markdown("### 🎯 Recognition Results")

                res_col1, res_col2 = st.columns([1, 1], gap="large")

                with res_col1:
                    st.markdown(
                        f"""
                        <div class="apple-card" style="text-align: center; padding: 2rem 1.5rem;">
                            <div style="font-size: 0.85rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                                Predicted Digit Class
                            </div>
                            <div class="digit-badge-hero">{pred_digit}</div>
                            <div style="margin-top: 0.8rem;">
                                <span class="status-pill-ok" style="font-size: 0.95rem; padding: 4px 12px;">
                                    {score_type}: {conf_pct:.1f}%
                                </span>
                            </div>
                            <div style="margin-top: 0.7rem; font-size: 0.84rem; color: #64748b;">
                                Classifier: <strong>Unified Ensemble Voting Classifier</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if "votes" in pred_res:
                        v = pred_res["votes"]
                        st.markdown(
                            f"""
                            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 10px; padding: 0.85rem 1rem; margin-top: 1rem; margin-bottom: 1rem; text-align: left;">
                                <div style="font-size: 0.82rem; font-weight: 700; color: #1e293b; text-transform: uppercase; margin-bottom: 0.4rem;">
                                    🗳️ Consensus Breakdown: {pred_res.get('consensus_type', '')}
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; font-size: 0.84rem; color: #334155;">
                                    <div style="background: #ffffff; padding: 6px; border-radius: 6px; text-align: center; border: 1px solid #e2e8f0;">
                                        <span style="color: #64748b; font-size: 0.76rem;">k-NN</span><br><strong>{v.get('knn', 'N/A')}</strong>
                                    </div>
                                    <div style="background: #ffffff; padding: 6px; border-radius: 6px; text-align: center; border: 1px solid #e2e8f0;">
                                        <span style="color: #64748b; font-size: 0.76rem;">Linear SVM</span><br><strong>{v.get('linear_svm', 'N/A')}</strong>
                                    </div>
                                    <div style="background: #ffffff; padding: 6px; border-radius: 6px; text-align: center; border: 1px solid #e2e8f0;">
                                        <span style="color: #64748b; font-size: 0.76rem;">RBF SVM</span><br><strong>{v.get('rbf_svm', 'N/A')}</strong>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("#### Top Alternative Candidates")
                    alt_cols = st.columns(3)
                    for idx, item in enumerate(pred_res["top_3"]):
                        with alt_cols[idx]:
                            st.markdown(
                                f"""
                                <div class="alt-candidate-card">
                                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">Rank #{item['rank']}</div>
                                    <div style="font-size: 1.6rem; font-weight: 700; color: #0f172a;">{item['digit']}</div>
                                    <div style="font-size: 0.82rem; color: #2563eb; font-weight: 600;">{item['percentage']:.1f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                with res_col2:
                    st.markdown("#### Class Score Distribution (0–9)")
                    all_scores = pred_res["all_scores"]
                    chart_data = pd.DataFrame({
                        "Digit": [str(d) for d in range(10)],
                        "Confidence (%)": [all_scores[str(d)] * 100 for d in range(10)]
                    })
                    st.bar_chart(chart_data.set_index("Digit"), color="#2563eb", height=220)

                    with st.expander("ℹ️ How was this confidence score calculated?"):
                        st.write(explanation)

                    with st.expander("🔬 View Preprocessed 28×28 Matrix (LeCun Centered)", expanded=False):
                        if processed_28x28 is not None:
                            pcol1, pcol2 = st.columns([1, 2])
                            with pcol1:
                                img_scaled = Image.fromarray((processed_28x28 * 255).astype(np.uint8)).resize((130, 130), Image.Resampling.NEAREST)
                                st.image(img_scaled, caption="28×28 Model Input", width=130)
                            with pcol2:
                                st.markdown(
                                    f"""
                                    - **Resolution:** 28 × 28 pixels
                                    - **Pixel Range:** [{processed_28x28.min():.2f}, {processed_28x28.max():.2f}]
                                    - **Center of Mass:** Centered at (14, 14)
                                    - **Aspect Ratio:** Preserved in 20×20 bounding box
                                    """
                                )


# ==============================================================================
# PAGE 3: MULTI-DIGIT RECOGNITION (DEDICATED PAGE)
# ==============================================================================
elif st.session_state["current_page"] == "Multi-Digit Recognition":
    st.markdown('<div class="page-badge">🔢 Multi-Digit Sequence Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Multi-Digit Number Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Automatic computer vision segmentation and sequential classification for handwritten multi-digit numbers (e.g., 42, 2026, 987654).</div>',
        unsafe_allow_html=True,
    )

    # Primary Sequence Engine Banner (Unified Ensemble)
    st.markdown(
        """
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.75rem 1.25rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.76rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">Primary Sequence Engine</span>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">Unified Ensemble Number Recognizer</div>
            </div>
            <span class="status-pill-ok" style="font-size: 0.84rem; padding: 4px 12px;">Segment-and-Vote Majority Consensus</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_key = "ensemble"

    input_tab1, input_tab2 = st.tabs([
        "✏️ Mode A: Draw Complete Number (Wide Canvas)",
        "📁 Mode B: Upload Number Image File"
    ])

    multi_image_input = None
    trigger_multi_prediction = False

    # TAB 1: DRAW COMPLETE NUMBER ON WIDE CANVAS
    with input_tab1:
        st.markdown("#### Draw an entire number (e.g. 45, 2026, 12345)")
        st.caption("Tip: Write numbers horizontally with natural spacing between digits.")

        col_draw_ctrl1, col_draw_ctrl2 = st.columns([1, 1])
        with col_draw_ctrl1:
            wide_canvas_mode = st.radio(
                "Canvas Style",
                ["White Stroke on Black Background", "Black Stroke on White Background"],
                horizontal=True,
                key="wide_canvas_mode"
            )
            if "White Stroke" in wide_canvas_mode:
                wide_stroke_color = "#FFFFFF"
                wide_bg_color = "#000000"
            else:
                wide_stroke_color = "#000000"
                wide_bg_color = "#FFFFFF"

        with col_draw_ctrl2:
            wide_stroke_width = st.slider("Pen Thickness", min_value=12, max_value=28, value=18, step=2, key="wide_stroke_width")

        if CANVAS_AVAILABLE:
            wide_canvas = render_st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=wide_stroke_width,
                stroke_color=wide_stroke_color,
                background_color=wide_bg_color,
                height=220,
                width=720,
                drawing_mode="freedraw",
                key="multi_digit_canvas_page",
            )
        else:
            st.error("streamlit-drawable-canvas is not installed.")
            wide_canvas = None

        if st.button("🔍 Recognize Drawn Number", type="primary", use_container_width=True, key="btn_rec_wide"):
            wide_img_data = safe_get_canvas_image(wide_canvas)
            if wide_img_data is not None:
                multi_image_input = wide_img_data
                trigger_multi_prediction = True
            else:
                st.warning("⚠️ Canvas is empty. Please draw a number.")

    # TAB 2: UPLOAD IMAGE OF HANDWRITTEN NUMBER
    with input_tab2:
        st.markdown("#### Upload an image of a handwritten number (PNG, JPG, JPEG)")
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=["png", "jpg", "jpeg"],
            key="multi_number_uploader"
        )

        if uploaded_file is not None:
            upload_img = Image.open(uploaded_file)
            st.image(upload_img, caption="Uploaded Original Image", width=360)

            if st.button("🔍 Recognize Uploaded Number", type="primary", key="btn_rec_upload_multi"):
                multi_image_input = upload_img
                trigger_multi_prediction = True

    # EXECUTE MULTI-DIGIT PREDICTION
    if trigger_multi_prediction and multi_image_input is not None:
        with st.spinner("Segmenting and classifying digits..."):
            res = predict_multi_digit_number(multi_image_input, model_key=selected_key)

        if not res.get("success"):
            st.warning(f"⚠️ {res.get('error', 'Recognition failed.')}")
        else:
            pred_num = res["predicted_number_str"]
            digit_cnt = res["digit_count"]
            avg_conf = res["average_confidence_percentage"]

            st.markdown("---")
            st.markdown("### 🎯 Sequence Recognition Output")

            # Prominent Result Display Banner
            st.markdown(
                f"""
                <div class="apple-card" style="text-align: center; padding: 2rem 1.5rem; background: #f8fafc;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">
                        Recognized Number Sequence
                    </div>
                    <div class="number-badge-hero">{pred_num}</div>
                    <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                        <span class="status-pill-ok" style="font-size: 0.88rem; padding: 5px 12px;">
                            Digits Count: {digit_cnt}
                        </span>
                        <span class="status-pill-ok" style="font-size: 0.88rem; padding: 5px 12px;">
                            Average Confidence: {avg_conf:.1f}%
                        </span>
                        <span style="background: #e2e8f0; color: #334155; font-size: 0.88rem; padding: 5px 12px; border-radius: 9999px; font-weight: 600;">
                            Classifier: Ensemble Voting Classifier
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if res.get("warning"):
                st.info(f"ℹ️ {res['warning']}")

            # Segmentation Visualization
            st.markdown("#### 🎯 Spatial Segmentation & Bounding Boxes")
            seg_col1, seg_col2 = st.columns(2)
            with seg_col1:
                st.image(res["annotated_image"], caption="Annotated Image with Left-to-Right Bounding Boxes", use_column_width=True)
            with seg_col2:
                st.image(res["binary_image"], caption="Binary Thresholded Image (White Foreground)", use_column_width=True)

            # Individual Digit Breakdown
            st.markdown("#### 🔬 Individual Digit Breakdown")
            digit_items = res["digits"]
            cols = st.columns(min(len(digit_items), 8))
            for idx, d in enumerate(digit_items):
                with cols[idx % len(cols)]:
                    st.markdown(f"**Digit #{d['index']}**")
                    img_scaled = Image.fromarray((d["crop_28x28"] * 255).astype(np.uint8)).resize((100, 100), Image.Resampling.NEAREST)
                    st.image(img_scaled, width=100)
                    st.markdown(
                        f"""
                        <div style="background: #2563eb; color: white; font-size: 1.8rem; font-weight: 700; border-radius: 8px; text-align: center; margin: 0.3rem 0;">
                            {d['predicted_digit']}
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b; text-align: center;">
                            Conf: <b>{d['confidence_percentage']:.1f}%</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Diagnostic Debug Panel if enabled
            if debug_mode:
                render_debug_panel(res, multi_image_input, debug_mode)


# ==============================================================================
# PAGE 4: DATASET EXPLORER
# ==============================================================================
elif st.session_state["current_page"] == "Dataset Explorer":
    st.markdown('<div class="page-badge">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">MNIST Benchmark Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore the 70,000-sample benchmark dataset: native image dimensions, class distributions, and spatial pixel density heatmaps.</div>',
        unsafe_allow_html=True,
    )

    try:
        X_all, y_all = get_cached_raw_mnist()
        data_loaded = True
    except Exception as e:
        st.warning(f"Could not load raw MNIST dataset: {e}")
        data_loaded = False

    if data_loaded:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Dataset Images", f"{X_all.shape[0]:,}")
        with m2:
            st.metric("Native Dimensions", "28 × 28 pixels")
        with m3:
            st.metric("Flattened Features", f"{X_all.shape[1]} dimensions")
        with m4:
            st.metric("Target Classes", "10 (Digits 0–9)")

        st.markdown("---")
        st.markdown("### 🖼️ Representative Digit Samples Grid")
        grid_plot_path = config.PLOTS_DIR / "sample_digits_grid.png"
        if grid_plot_path.exists():
            st.image(str(grid_plot_path), caption="Representative Digit Samples from the MNIST Dataset", use_column_width=True)

        st.markdown("---")
        col_dist, col_heatmap = st.columns(2)

        with col_dist:
            st.markdown("### 📊 Class Distribution")
            unique, counts = np.unique(y_all, return_counts=True)
            dist_df = pd.DataFrame({"Digit": [str(u) for u in unique], "Sample Count": counts})
            st.bar_chart(dist_df.set_index("Digit"), color="#2563eb", height=260)
            st.caption("The MNIST dataset features an exceptionally balanced class distribution (~6,000–7,000 samples per class).")

        with col_heatmap:
            st.markdown("### 🔥 Average Pixel Intensity Heatmap")
            selected_digit_for_heatmap = st.selectbox("Select Digit for Heatmap Analysis:", list(range(10)), index=0)
            fig_hm = plot_average_digit_intensity(X_all[:10000] / 255.0, y_all[:10000], selected_digit_for_heatmap)
            st.pyplot(fig_hm)
            st.caption("Bright regions highlight the most common stroke pathways drawn for this digit class across thousands of writers.")

        st.markdown(
            """
            <div class="apple-card" style="margin-top: 1.5rem;">
                <h3 style="margin-top: 0; font-size: 1.15rem; font-weight: 600; color: #0f172a;">💡 How 2D Images Become 1D Feature Vectors</h3>
                <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                    Each digit is a <b>28 × 28 pixel 2D matrix</b> where every cell holds an intensity from 0 (background) to 255 (white stroke).
                    To feed this into standard classical classifiers (k-NN and SVM), we unroll (flatten) the 28 rows into a single 
                    continuous <b>784-dimensional numerical feature vector</b>:
                </p>
                <code style="background: #f1f5f9; padding: 4px 8px; border-radius: 6px; color: #0f172a; font-size: 0.9rem;">
                    x = [p_0, p_1, p_2, ..., p_783]  where p_i ∈ [0.0, 1.0]
                </code>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 5: ABOUT SYSTEM & METHODOLOGY (PROJECT GUIDE & VIVA)
# ==============================================================================
elif st.session_state["current_page"] == "About System":
    st.markdown('<div class="page-badge">📖 Academic Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Ensemble System Methodology & Reference Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed theoretical foundations, mathematical formulations, internal evaluation benchmarks, and viva examination preparation.</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🎯 Objectives & Problem",
        "🧠 Mathematical Formulations",
        "⚙️ Preprocessing Protocol",
        "🔢 Multi-Digit Segmentation",
        "📊 Internal System Evaluation",
        "🎓 Viva Q&A Master Guide",
    ])

    with t1:
        st.markdown(
            """
            ### Problem Statement
            Handwritten digit recognition poses significant computer vision challenges due to diverse handwriting styles,
            variable stroke widths, slant angles, and spatial shifts. The objective is to design, implement, and evaluate
            a machine learning system that accurately classifies isolated handwritten digits (0–9) and variable-length numbers
            using a unified ensemble of classical algorithms without requiring deep neural networks.

            ### Project Objectives
            1. **Ensemble Architecture:** Implement a democratic Hard Majority Voting Classifier combining k-NN, Linear SVM, and RBF Kernel SVM with deterministic tie-breaking.
            2. **Zero-Leakage Experimental Design:** Enforce strict separation between training hyperparameter selection and held-out test evaluation.
            3. **Aspect-Ratio & Mass Preprocessing:** Build an image normalization pipeline adhering to Yann LeCun's center-of-mass centering standard.
            4. **Real-Time Interactive Interface:** Provide an interactive drawing canvas and file upload with unified ensemble inference.
            5. **End-to-End Multi-Digit Recognition:** Implement automated segmentation of multi-digit numbers via contour analysis and vertical projection profile splitting.
            """
        )

    with t2:
        st.markdown(
            r"""
            ### 1. k-Nearest Neighbors (k-NN)
            - **Type:** Non-parametric, lazy instance-based learning.
            - **Decision Rule:** For a query point $\mathbf{x}$, locate the $k$ nearest training points using distance $d(\mathbf{x}, \mathbf{x}_i)$. The predicted class is determined by majority vote or distance-weighted votes:
            $$\hat{y} = \arg\max_c \sum_{i \in N_k(\mathbf{x}), y_i = c} w_i, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$$
            - **Pros & Cons:** No training phase ($O(1)$), but memory-intensive and slower inference for large $N$.

            ### 2. Linear Support Vector Machine (Linear SVM)
            - **Type:** Maximum margin linear classifier.
            - **Optimization Objective:** Find the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ that maximizes the functional margin between classes while penalizing margin violations with parameter $C$:
            $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 - \xi_i$$
            - **Pros & Cons:** Extremely fast training with LIBLINEAR, sub-millisecond inference, but limited on nonlinearly separable patterns.

            ### 3. Support Vector Machine with RBF Kernel (RBF SVM)
            - **Type:** Kernelized maximum margin classifier.
            - **The Kernel Trick:** Implicitly projects 784-dimensional features into an infinite-dimensional reproducing kernel Hilbert space (RKHS) using Gaussian Radial Basis Functions:
            $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
            - **Pros & Cons:** Superior classification accuracy (>98%) on complex curved strokes, but training scales as $O(N^2)$ to $O(N^3)$.

            ### 4. Ensemble Voting Classifier (Hard Majority Voting)
            - **Type:** Meta-classifier combining heterogeneous learning paradigms (Instance-based + Linear margin + Non-linear kernel).
            - **Decision Rule (Hard Voting):** Each of the $M=3$ base classifiers casts a single discrete vote $\hat{y}_m = h_m(\mathbf{x}) \in \{0, \dots, 9\}$:
            $$\hat{y}_{\text{ensemble}} = \arg\max_{c \in \{0,\dots,9\}} \sum_{m=1}^{M} \mathbb{I}(h_m(\mathbf{x}) = c)$$
            - **Deterministic Tie-Breaking:** In the event of a 1-1-1 three-way tie where each base model votes for a different class, the decision is resolved using RBF SVM (the highest-accuracy individual baseline).
            - **Trade-off Insights:** Achieves high stability and transparent agreement diagnostics (3/3 Unanimous vs 2/3 Majority). Unweighted majority voting achieves 96.05% accuracy on held-out test data.
            """
        )

    with t3:
        st.markdown(
            r"""
            ### Standard MNIST Preprocessing Pipeline
            Our application applies the exact normalization steps defined by Yann LeCun in 1998:
            1. **Polarity Check & Inversion:** Invert user canvas (black-on-white) to MNIST standard (white stroke on black background).
            2. **Bounding Box Isolation:** Detect non-zero bounding box $[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$ around the digit stroke.
            3. **Aspect Ratio Preservation:** Resize the bounding box to fit inside a $20 \times 20$ pixel square without distorting proportions.
            4. **$28 \times 28$ Canvas Placement:** Center the $20 \times 20$ image onto a blank $28 \times 28$ canvas.
            5. **Center-of-Mass Correction:** Compute the center-of-mass $(\bar{x}, \bar{y})$ using pixel intensities and shift the image so the center-of-mass aligns with $(14, 14)$.
            6. **Intensity Normalization:** Divide pixel intensities by 255.0 to yield values in $[0.0, 1.0]$.
            """
        )

    with t4:
        st.markdown(
            r"""
            ### Multi-Digit Number Segmentation Pipeline
            Single-digit MNIST models cannot classify multi-digit strings directly. Our system bridges this gap
            via an automated computer vision segmentation pipeline:

            1. **Otsu Binarization & Polarity Normalization:**
               The drawing or uploaded image is converted to grayscale, background polarity is checked (inverting dark-on-light ink), and Otsu thresholding produces a clean binary mask.
            
            2. **Connected-Component & Contour Analysis:**
               `cv2.findContours` identifies candidate digit regions. Specks with pixel area $< 40$ are eliminated as noise.

            3. **Multi-Stroke / Disjoint Component Merging:**
               Digits composed of disconnected strokes (e.g. cross on 7, top bar of 5, detached strokes of 4) are merged if their bounding boxes overlap horizontally by $> 50\%$ or lie in close vertical proximity.

            4. **Touching Digit Splitting via Vertical Projection Profile:**
               When handwriting joins adjacent digits, the combined bounding box has an abnormal aspect ratio ($w / h > 1.25$). The system computes the column-wise vertical projection profile:
               $$\text{Profile}(x) = \sum_{y} I(x, y)$$
               and locates the deepest valley between 25% and 75% of the width to cleanly bifurcate touching digits.

            5. **Spatial Sequencing & Normalization:**
               Detected bounding boxes are strictly sorted left-to-right by $x_{\min}$. Each box is cropped, aspect-ratio resized to $20 \times 20$, centered by intensity mass on $28 \times 28$, and classified independently before concatenation.
            """
        )

    with t5:
        st.markdown("### 📊 Internal System Evaluation (Held-Out MNIST Test Set)")
        st.caption(
            "Evaluation Protocol: Evaluated on 2,000 held-out test samples strictly partitioned from Yann LeCun's official MNIST test pool. Zero test data leakage."
        )

        metrics_json = get_cached_metrics()
        ens_metrics = metrics_json.get("ensemble", {}) if metrics_json else {}

        em1, em2, em3, em4 = st.columns(4)
        with em1:
            st.metric("Test Accuracy", f"{ens_metrics.get('accuracy', 0.9605) * 100:.2f}%")
        with em2:
            st.metric("Macro Precision", f"{ens_metrics.get('macro_precision', 0.9607) * 100:.2f}%")
        with em3:
            st.metric("Macro Recall", f"{ens_metrics.get('macro_recall', 0.9598) * 100:.2f}%")
        with em4:
            st.metric("Macro F1-Score", f"{ens_metrics.get('macro_f1', 0.9601) * 100:.2f}%")

        st.markdown("---")
        col_cm, col_pc = st.columns([1.2, 1])
        with col_cm:
            st.markdown("#### Ensemble Confusion Matrix")
            cm_img_path = config.PLOTS_DIR / "confusion_matrix_ensemble.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), caption="Ensemble Voting Classifier Confusion Matrix (Test Set)", use_column_width=True)
            else:
                st.caption("Ensemble confusion matrix plot not found.")

        with col_pc:
            st.markdown("#### Per-Class Performance Breakdown")
            if "per_class" in ens_metrics:
                per_class_df = pd.DataFrame(ens_metrics["per_class"]).T
                per_class_df.index.name = "Digit"
                st.dataframe(per_class_df, use_container_width=True)
            else:
                st.caption("Per-class metrics not available.")

        st.markdown(
            """
            <div class="apple-card" style="margin-top: 1rem;">
                <h4 style="margin-top: 0; font-size: 1rem; font-weight: 600; color: #0f172a;">🔬 Evaluation Methodology & Leakage Prevention</h4>
                <p style="color: #475569; font-size: 0.88rem; line-height: 1.55; margin-bottom: 0;">
                    To prevent data leakage, training and hyperparameter selection were performed solely on a stratified training partition (10,000 samples) via 3-Fold Stratified Cross-Validation.
                    The ensemble combines k-NN (k=3), calibrated Linear SVM, and RBF Kernel SVM via hard majority voting.
                    Evaluation was executed once on the untouched held-out 2,000 test images.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t6:
        st.markdown("### Top Viva Questions & Model Answers")

        viva_items = [
            (
                "Q1: What is the Kernel Trick in Support Vector Machines?",
                "The kernel trick allows an SVM to compute inner products in an arbitrary high-dimensional feature space without explicitly calculating the high-dimensional coordinates of the data points. A kernel function K(x, z) computes the dot product ⟨Φ(x), Φ(z)⟩ directly in the original input space, making nonlinear classification computationally tractable."
            ),
            (
                "Q2: Why do we normalize pixel intensities from [0, 255] to [0, 1]?",
                "Distance-based classifiers like k-NN and SVM rely on Euclidean distance ||x_1 - x_2|| or dot products. Normalizing pixel intensities to [0, 1] prevents numerical overflow, ensures uniform gradient updates during optimization, and standardizes feature scale across all 784 dimensions."
            ),
            (
                "Q3: What is the role of hyperparameters C and gamma in RBF SVM?",
                "Parameter C balances maximizing the decision margin against minimizing classification errors (high C creates a narrower margin with lower bias; low C allows softer margins with higher regularization). Gamma determines the reach of a single training instance (high gamma means each instance has a tight local influence, risking overfitting; low gamma creates smooth decision boundaries)."
            ),
            (
                "Q4: What is the difference between Macro F1 and Weighted F1?",
                "Macro F1 computes the unweighted arithmetic mean of F1-scores across all classes, giving equal importance to every class regardless of sample count. Weighted F1 weights each class score by its support (frequency in the dataset). For balanced datasets like MNIST, they are typically close."
            ),
            (
                "Q5: How does k-NN handle inference latency vs SVM?",
                "k-NN is a lazy learner: it performs zero computation during training, but during prediction it must calculate distances to all N training instances, resulting in O(N × D) inference latency. SVM is an eager learner: it spends time during training to identify support vectors, but inference requires evaluating only the support vectors (or a single w · x dot product for Linear SVM), resulting in sub-millisecond predictions."
            ),
            (
                "Q6: Why is Center of Mass centering crucial for MNIST recognition?",
                "If a digit is merely placed in the bounding box center, variations in stroke distribution (e.g. the long vertical stroke in '1' vs the circular loop in '0') cause spatial misalignment with the training templates. Shifting the center-of-mass of pixel intensity to (14, 14) guarantees uniform spatial alignment with the standard MNIST distribution."
            ),
            (
                "Q7: How does your system recognize multi-digit numbers using single-digit models?",
                "Our system employs a computer vision pipeline that performs Otsu binarization, contour-based connected component extraction, heuristic merging of broken/multi-stroke components (like 4, 5, 7), vertical projection profile valley-splitting for touching digits, and left-to-right spatial sorting. Each segmented bounding box is then normalized into a 28×28 LeCun image and classified sequentially."
            ),
            (
                "Q8: How are confidence scores computed for models that don't output probabilities?",
                "For k-NN, confidence is computed as the distance-weighted proportion of votes belonging to the winning class among the k nearest neighbors. For Linear SVM and RBF SVM, decision function distances (signed geometric margins) are passed through a calibrated Softmax/Platt transformation to map margins into well-behaved probabilities summing to 1.0."
            ),
            (
                "Q9: How does the Ensemble Voting Classifier decide the final prediction?",
                "It employs hard majority voting: each of the three base models (k-NN, Linear SVM, RBF SVM) casts one vote. If at least 2 models agree (unanimous 3/3 or majority 2/3), that consensus class is predicted. If all three models predict different digits (1/1/1 split tie), the tie is deterministically broken by RBF SVM because it demonstrated the highest individual baseline test accuracy."
            ),
            (
                "Q10: Why did the Ensemble Voting Classifier achieve 96.05% test accuracy while RBF SVM achieved 96.55%?",
                "Ensemble performance depends on base classifier diversity and error independence. When a weaker baseline like Linear SVM (90.55% accuracy) and k-NN (94.60% accuracy) happen to make correlated errors on difficult or ambiguous stroke patterns, their combined 2 votes can overrule the correct prediction of RBF SVM (96.55%). Rather than outperforming the top model on raw score, the ensemble's primary value is eliminating single-model blind spots, variance reduction, and providing multi-model agreement diagnostics."
            ),
        ]

        for q, a in viva_items:
            with st.expander(f"❓ {q}"):
                st.write(a)
