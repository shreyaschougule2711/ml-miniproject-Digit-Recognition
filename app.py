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

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: var(--color-text-main);
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
        background: var(--color-primary-soft);
        color: var(--color-primary);
        border: 1px solid var(--color-primary-border);
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
        color: var(--color-text-main);
        letter-spacing: -0.025em;
        line-height: 1.25;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: var(--color-text-sub);
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
        color: #0f172a;
        margin-bottom: 0.25rem;
    }

    .workflow-desc {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.4;
    }

    /* Model Cards */
    .model-card {
        background: #ffffff;
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
        color: #0f172a;
        margin-bottom: 0.4rem;
    }

    .model-card-desc {
        font-size: 0.86rem;
        color: #64748b;
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
        color: #0f172a;
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

        st.markdown("#### Cross-Classifier Predictions (Same Input)")
        rows = []
        for d in digit_items:
            flat = d["crop_28x28"].reshape(1, 784).astype(np.float32)
            row = {"Digit #": d["index"]}
            for mk, label in [("knn", "k-NN"), ("linear_svm", "Linear SVM"), ("rbf_svm", "RBF SVM")]:
                pred_res = predict_digit(mk, flat)
                if pred_res.get("success"):
                    row[f"{label} Pred"] = pred_res["predicted_digit"]
                    row[f"{label} Conf%"] = f"{pred_res['confidence_percentage']:.1f}%"
                else:
                    row[f"{label} Pred"] = "N/A"
                    row[f"{label} Conf%"] = "N/A"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==============================================================================
# SIDEBAR APPLICATION SHELL & NAVIGATION
# ==============================================================================
nav_options = [
    "Dashboard",
    "Recognize Digits",
    "Multi-Digit Recognition",
    "Model Comparison",
    "Dataset Explorer",
    "About Project",
]

# Synchronize session state navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

def on_nav_change():
    st.session_state["current_page"] = st.session_state["nav_radio_select"]

# Sidebar Header Branding
st.sidebar.markdown(
    """
    <div style="padding-top: 0.5rem; margin-bottom: 0.2rem;">
        <div class="sidebar-brand-title">
            <span>✍️ DigitAI</span>
            <span class="sidebar-brand-badge">STUDIO</span>
        </div>
        <div class="sidebar-caption">Handwritten Digit & Number Recognition</div>
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
for key, is_avail in avail_status.items():
    name = {
        "knn": "k-NN Classifier",
        "linear_svm": "Linear SVM",
        "rbf_svm": "RBF Kernel SVM",
    }.get(key, key)
    badge = '<span class="status-pill-ok">● Ready</span>' if is_avail else '<span class="status-pill-missing">○ Missing</span>'
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 0.84rem;">
            <span style="color: #334155; font-weight: 500;">{name}</span>
            <span>{badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not all(avail_status.values()):
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
    st.markdown('<div class="page-badge">✨ Supervised Machine Learning Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Handwritten Digit Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">An academic machine learning system rigorously comparing k-NN, Linear SVM, and RBF Kernel SVM on the landmark MNIST benchmark dataset.</div>',
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
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 600; color: #0f172a;">Live Recognition Studio</h3>
                </div>
                <p style="font-size: 0.88rem; color: #64748b; line-height: 1.45; margin-bottom: 1rem;">
                    Draw custom digits on an interactive canvas or upload handwritten images to test real-time classification across trained models.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("👉 Open Digit Recognition Studio", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Recognize Digits"
            st.rerun()

    with col_act2:
        st.markdown(
            """
            <div class="apple-card" style="margin-bottom: 0.75rem;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.4rem;">📊</span>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 600; color: #0f172a;">Evaluation Benchmarks</h3>
                </div>
                <p style="font-size: 0.88rem; color: #64748b; line-height: 1.45; margin-bottom: 1rem;">
                    Inspect confusion matrices, precision/recall breakdowns, and inference latency benchmarks evaluated on the untouched MNIST test set.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📈 Explore Model Performance", use_container_width=True):
            st.session_state["current_page"] = "Model Comparison"
            st.rerun()

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
                <div class="metric-label">Features (28×28 Grid)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        meta = get_cached_metadata()
        best_acc = "98.1%" if not meta else f"{meta['results_summary'].get('rbf_svm', {}).get('accuracy', 0.98)*100:.1f}%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{best_acc}</div>
                <div class="metric-label">Top Test Accuracy</div>
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
                <div class="workflow-title">Supervised Inference</div>
                <div class="workflow-desc">Pass 784-D vector to k-NN spatial voting or SVM hyperplane decision function.</div>
            </div>
            <div class="workflow-step">
                <span class="workflow-num">Step 5</span>
                <div class="workflow-title">Prediction & Analytics</div>
                <div class="workflow-desc">Output predicted digit, calibrated confidence score, and top candidate distribution.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Machine Learning Classifiers Profile
    st.markdown("### 🧠 Evaluated Machine Learning Paradigms")
    mcol1, mcol2, mcol3 = st.columns(3)

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
                    <div class="model-spec-row"><span>Inference Latency</span><strong>O(N · D)</strong></div>
                    <div class="model-spec-row"><span>Key Hyperparameter</span><strong>k = 3, Metric = L2</strong></div>
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
                    <div class="model-spec-row"><span>Inference Latency</span><strong>Sub-millisecond O(D)</strong></div>
                    <div class="model-spec-row"><span>Key Hyperparameter</span><strong>C = 0.1, L2 Penalty</strong></div>
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
                        <span style="font-size: 0.8rem; font-weight: 600; color: #1d4ed8;">Top Performer</span>
                    </div>
                    <div class="model-card-title">RBF Kernel SVM</div>
                    <div class="model-card-desc">
                        Projects 784-D pixel vectors into an infinite-dimensional Hilbert space using the Gaussian Radial Basis Function kernel to capture complex stroke curves.
                    </div>
                </div>
                <div>
                    <div class="model-spec-row"><span>Training Complexity</span><strong>O(N² ~ N³)</strong></div>
                    <div class="model-spec-row"><span>Test Accuracy</span><strong>98.1% (Peak)</strong></div>
                    <div class="model-spec-row"><span>Key Hyperparameter</span><strong>C = 5.0, γ = scale</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 2: RECOGNIZE DIGITS (SINGLE DIGIT STUDIO)
# ==============================================================================
elif st.session_state["current_page"] == "Recognize Digits":
    st.markdown('<div class="page-badge">✍️ Single-Digit Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Single Digit Recognition Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Draw an isolated digit (0–9) or upload a handwriting sample to test real-time classification across trained models.</div>',
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

    # Model Selector
    model_col1, model_col2 = st.columns([2, 1])
    with model_col1:
        model_choice = st.selectbox(
            "Select Machine Learning Classifier:",
            [
                ("rbf_svm", "RBF Kernel Support Vector Machine (Highest Non-linear Accuracy)"),
                ("knn", "k-Nearest Neighbors (Instance-Based Spatial Voting)"),
                ("linear_svm", "Linear Support Vector Machine (Fast Hyperplane Margin)"),
            ],
            format_func=lambda x: x[1],
        )
        selected_key = model_choice[0]

    with model_col2:
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.65rem 1rem; margin-top: 1.5rem;">
                <span style="font-size: 0.8rem; color: #64748b;">Active Classifier:</span><br>
                <strong style="color: #0f172a; font-size: 0.92rem;">{model_choice[1].split('(')[0].strip()}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

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
                canvas_result = st_canvas(
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
                if canvas_result is not None and getattr(canvas_result, "image_data", None) is not None:
                    active_image_input = canvas_result.image_data
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
                                Classifier: <strong>{model_choice[1].split('(')[0].strip()}</strong>
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

                # Cross-Classifier Evaluation Comparison for this Single Digit
                st.markdown("#### ⚖️ Cross-Classifier Consensus on this Input")
                cross_rows = []
                for mk, label in [("knn", "k-NN"), ("linear_svm", "Linear SVM"), ("rbf_svm", "RBF SVM")]:
                    pres = predict_digit(mk, flat_784)
                    if pres.get("success"):
                        cross_rows.append({
                            "Classifier": label,
                            "Predicted Digit": pres["predicted_digit"],
                            "Confidence": f"{pres['confidence_percentage']:.1f}%",
                            "Score Metric": pres["score_type"],
                        })
                st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)


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

    # Model Selector
    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        model_choice = st.selectbox(
            "Select Machine Learning Classifier for Sequence:",
            [
                ("rbf_svm", "RBF Kernel Support Vector Machine (Highest Non-linear Accuracy)"),
                ("knn", "k-Nearest Neighbors (Instance-Based Spatial Voting)"),
                ("linear_svm", "Linear Support Vector Machine (Fast Hyperplane Margin)"),
            ],
            format_func=lambda x: x[1],
            key="multi_model_select",
        )
        selected_key = model_choice[0]

    with m_col2:
        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.65rem 1rem; margin-top: 1.5rem;">
                <span style="font-size: 0.8rem; color: #64748b;">Active Sequence Classifier:</span><br>
                <strong style="color: #0f172a; font-size: 0.92rem;">{model_choice[1].split('(')[0].strip()}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

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
            wide_canvas = st_canvas(
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
            if wide_canvas is not None and getattr(wide_canvas, "image_data", None) is not None:
                multi_image_input = wide_canvas.image_data
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
                            Classifier: {model_choice[1].split('(')[0].strip()}
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

            # Cross-Classifier Consensus on Digits
            st.markdown("#### ⚖️ Cross-Classifier Consensus for Sequence")
            seq_rows = []
            for d in digit_items:
                flat = d["crop_28x28"].reshape(1, 784).astype(np.float32)
                row = {"Position": f"Digit #{d['index']}"}
                for mk, label in [("knn", "k-NN"), ("linear_svm", "Linear SVM"), ("rbf_svm", "RBF SVM")]:
                    p_res = predict_digit(mk, flat)
                    if p_res.get("success"):
                        row[f"{label} Pred"] = p_res["predicted_digit"]
                        row[f"{label} Conf"] = f"{p_res['confidence_percentage']:.1f}%"
                    else:
                        row[f"{label} Pred"] = "N/A"
                        row[f"{label} Conf"] = "N/A"
                seq_rows.append(row)
            st.dataframe(pd.DataFrame(seq_rows), use_container_width=True, hide_index=True)

            # Diagnostic Debug Panel if enabled
            if debug_mode:
                render_debug_panel(res, multi_image_input, debug_mode)


# ==============================================================================
# PAGE 4: MODEL COMPARISON & BENCHMARKS
# ==============================================================================
elif st.session_state["current_page"] == "Model Comparison":
    st.markdown('<div class="page-badge">📊 Rigorous Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Model Comparison & Evaluation Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Authentic performance metrics calculated on the untouched MNIST test set with zero data leakage.</div>',
        unsafe_allow_html=True,
    )

    comp_df = get_cached_comparison_table()
    metrics_json = get_cached_metrics()
    meta = get_cached_metadata()

    if comp_df is None or metrics_json is None:
        st.info("No saved evaluation metrics found yet. Run training to generate authentic benchmark results.")
        if st.button("🚀 Run Training & Benchmark Pipeline Now", type="primary"):
            with st.spinner("Training models and computing evaluation metrics..."):
                from src.train import run_training_pipeline
                run_training_pipeline(train_samples=5000, test_samples=1500, tune=False)
                st.success("Evaluation completed!")
                st.cache_data.clear()
                st.rerun()
    else:
        # Display dataset configuration alert
        if meta and "dataset" in meta:
            dmeta = meta["dataset"]
            st.caption(
                f"Evaluation Protocol: Train size = {dmeta['train_samples_used']:,} instances "
                f"(Subset: {dmeta['is_train_subset']}), Test size = {dmeta['test_samples_used']:,} instances "
                f"(Subset: {dmeta['is_test_subset']}). Execution Date: {meta.get('execution_date', 'N/A')}"
            )

        # Comparative Table
        st.markdown("### 🏆 Performance Summary Table")
        display_df = comp_df.copy()
        for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")
        if "Training Time (s)" in display_df.columns:
            display_df["Training Time (s)"] = display_df["Training Time (s)"].apply(lambda x: f"{x:.2f}s")
        if "Inference Latency (ms/sample)" in display_df.columns:
            display_df["Inference Latency (ms/sample)"] = display_df["Inference Latency (ms/sample)"].apply(lambda x: f"{x:.2f} ms")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Charts Section
        st.markdown("### 📈 Visual Benchmark Charts")
        chart_col1, chart_col2 = st.columns(2)

        chart1_path = config.PLOTS_DIR / "model_metrics_comparison.png"
        chart2_path = config.PLOTS_DIR / "runtime_comparison.png"

        with chart_col1:
            if chart1_path.exists():
                st.image(str(chart1_path), caption="Test Accuracy vs Macro F1-Score Comparison", use_column_width=True)
            else:
                st.caption("Bar chart plot not found.")

        with chart_col2:
            if chart2_path.exists():
                st.image(str(chart2_path), caption="Training Duration & Inference Latency Benchmark", use_column_width=True)
            else:
                st.caption("Runtime comparison plot not found.")

        # Confusion Matrix Inspector
        st.markdown("### 🎯 Confusion Matrix Inspector")
        cm_model_choice = st.selectbox(
            "Select Model to Inspect Confusion Matrix:",
            [
                ("rbf_svm", "RBF Kernel Support Vector Machine"),
                ("knn", "k-Nearest Neighbors"),
                ("linear_svm", "Linear Support Vector Machine"),
            ],
            format_func=lambda x: x[1],
        )
        selected_cm_key = cm_model_choice[0]
        cm_path = config.PLOTS_DIR / f"confusion_matrix_{selected_cm_key}.png"

        col_cm_plot, col_cm_insights = st.columns([1.2, 1])
        with col_cm_plot:
            if cm_path.exists():
                st.image(str(cm_path), caption=f"Confusion Matrix ({cm_model_choice[1]})", use_column_width=True)
            else:
                st.caption("Confusion matrix plot file not found.")

        with col_cm_insights:
            st.markdown("#### Per-Class Performance Breakdown")
            if selected_cm_key in metrics_json:
                per_class = metrics_json[selected_cm_key].get("per_class", {})
                per_class_df = pd.DataFrame(per_class).T
                per_class_df.index.name = "Digit"
                st.dataframe(per_class_df, use_container_width=True)

        # Model Trade-offs Explanation
        st.markdown(
            r"""
            <div class="apple-card" style="margin-top: 1.5rem;">
                <h3 style="margin-top: 0; font-size: 1.15rem; font-weight: 600;">⚖️ Architectural Model Trade-off Analysis</h3>
                <ul style="color: #475569; font-size: 0.92rem; line-height: 1.6; padding-left: 1.2rem;">
                    <li><b>RBF Kernel SVM:</b> Delivers the highest overall accuracy and F1-score because the Gaussian kernel can map nonlinear stroke curvatures into high-dimensional Hilbert space. However, training time scales quadratically with dataset size $O(N^2)$ to $O(N^3)$.</li>
                    <li><b>k-NN:</b> Zero training time ($O(1)$ lazy fit), but prediction latency scales with training dataset size $O(N \cdot D)$ because distances must be calculated to every training sample during prediction.</li>
                    <li><b>Linear SVM:</b> Very fast training with LIBLINEAR and sub-millisecond inference latency ($O(D)$ dot product). However, accuracy is bounded because certain digit pairs (e.g. 4 vs 9, 3 vs 5) are not linearly separable in raw pixel space.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Multi-Digit Sequence Benchmark Section
        multi_csv_path = config.RESULTS_DIR / "multidigit_benchmark.csv"
        if multi_csv_path.exists():
            st.markdown("### 🔢 Multi-Digit Sequence Evaluation Benchmark")
            mdf = pd.read_csv(multi_csv_path)

            mb1, mb2, mb3, mb4 = st.columns(4)
            with mb1:
                st.metric("Sequences Tested", f"{len(mdf)}")
            with mb2:
                seg_succ = (mdf["Segmentation Success"].sum() / len(mdf)) * 100
                st.metric("Segmentation Success", f"{seg_succ:.1f}%")
            with mb3:
                exact_succ = (mdf["Exact Match"].sum() / len(mdf)) * 100
                st.metric("Exact Match Rate", f"{exact_succ:.1f}%")
            with mb4:
                st.metric("Tested Lengths", f"{mdf['Length'].min()}–{mdf['Length'].max()} digits")

            with st.expander("📋 View Evaluated Multi-Digit Sequences Sample Table", expanded=False):
                st.dataframe(
                    mdf[["Target Number", "Predicted Number", "Length", "Segmentation Success", "Exact Match", "Confidence (%)"]].head(25),
                    use_container_width=True,
                    hide_index=True,
                )


# ==============================================================================
# PAGE 5: DATASET EXPLORER
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
                <h3 style="margin-top: 0; font-size: 1.15rem; font-weight: 600;">💡 How 2D Images Become 1D Feature Vectors</h3>
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
# PAGE 6: ABOUT & METHODOLOGY (PROJECT GUIDE & VIVA)
# ==============================================================================
elif st.session_state["current_page"] == "About Project":
    st.markdown('<div class="page-badge">📖 Academic Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Project Methodology & Viva Reference Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed theoretical foundations, mathematical formulations, algorithmic comparisons, and viva examination preparation.</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5 = st.tabs([
        "🎯 Objectives & Problem",
        "🧠 Mathematical Formulations",
        "⚙️ Preprocessing Protocol",
        "🔢 Multi-Digit Segmentation",
        "🎓 Viva Q&A Master Guide",
    ])

    with t1:
        st.markdown(
            """
            ### Problem Statement
            Handwritten digit recognition poses significant computer vision challenges due to diverse handwriting styles,
            variable stroke widths, slant angles, and spatial shifts. The objective is to design, implement, and evaluate
            a machine learning system that accurately classifies isolated handwritten digits (0–9) without requiring deep neural networks.

            ### Project Objectives
            1. **Supervised Classifier Comparison:** Implement and rigorously evaluate k-NN, Linear SVM, and RBF Kernel SVM.
            2. **Zero-Leakage Experimental Design:** Enforce strict separation between training hyperparameter selection and held-out test evaluation.
            3. **Aspect-Ratio & Mass Preprocessing:** Build an image normalization pipeline adhering to LeCun's center-of-mass centering standard.
            4. **Real-Time Interactive Interface:** Provide an interactive drawing canvas with real-time inference and transparent score interpretations.
            5. **End-to-End Multi-Digit Recognition:** Implement automated segmentation of multi-digit strings via contour analysis and vertical projection profile splitting.
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
        ]

        for q, a in viva_items:
            with st.expander(f"❓ {q}"):
                st.write(a)
