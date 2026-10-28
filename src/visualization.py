"""
Visualization Utilities for Handwritten Digit Recognition System.

Generates high-quality publication-ready figures for:
- Confusion matrices
- Accuracy & F1-score model comparisons
- Training vs inference runtime benchmarks
- Sample digit grids and pixel intensity heatmaps
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Set modern aesthetic theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: List[str] = [str(i) for i in range(10)],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Renders and optionally saves an annotated heatmap confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#f0f0f0"
    )
    ax.set_title(f"Confusion Matrix: {model_name}", pad=14, fontweight="bold")
    ax.set_xlabel("Predicted Digit Label", labelpad=10)
    ax.set_ylabel("True Ground-Truth Digit", labelpad=10)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualization] Saved confusion matrix to {save_path}")

    return fig


def plot_metrics_bar_chart(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots side-by-side comparative bar chart of Accuracy and Macro F1-Score.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comparison_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, comparison_df["Accuracy"] * 100, width, label="Test Accuracy (%)", color="#2563eb")
    rects2 = ax.bar(x + width/2, comparison_df["Macro F1"] * 100, width, label="Macro F1-Score (%)", color="#0d9488")

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Classifier Performance Benchmark on Test Set", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], fontweight="semibold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
    ax.set_ylim(80, 102)

    # Attach score values above bars
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualization] Saved metrics bar chart to {save_path}")

    return fig


def plot_runtime_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualizes training time and test inference latency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Training Time
    colors = ["#3b82f6", "#10b981", "#8b5cf6"]
    bars1 = ax1.bar(comparison_df["Model"], comparison_df["Training Time (s)"], color=colors, width=0.5)
    ax1.set_ylabel("Seconds", fontweight="bold")
    ax1.set_title("Training Duration (CPU)", fontweight="bold")
    for bar in bars1:
        val = bar.get_height()
        ax1.annotate(f"{val:.2f}s",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    # Inference Latency (ms per 100 samples)
    if "Inference Latency (ms/sample)" in comparison_df.columns:
        ms_per_sample = comparison_df["Inference Latency (ms/sample)"]
    else:
        ms_per_sample = comparison_df["Prediction Time (s)"] / 10.0  # approximate scaling
    bars2 = ax2.bar(comparison_df["Model"], ms_per_sample, color=colors, width=0.5)
    ax2.set_ylabel("Milliseconds per Sample", fontweight="bold")
    ax2.set_title("Inference Latency (Single Digit)", fontweight="bold")
    for bar in bars2:
        val = bar.get_height()
        ax2.annotate(f"{val:.2f} ms",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualization] Saved runtime comparison chart to {save_path}")

    return fig


def plot_sample_digit_grid(
    X: np.ndarray,
    y: np.ndarray,
    num_cols: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots an exemplary grid showing one representative sample for each digit 0-9.
    """
    fig, axes = plt.subplots(1, 10, figsize=(12, 2.2))
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        img = X[idx].reshape(28, 28)
        axes[digit].imshow(img, cmap="gray")
        axes[digit].set_title(f"Label: {digit}", fontsize=11, fontweight="bold")
        axes[digit].axis("off")

    plt.suptitle("MNIST Dataset Sample Images (0 to 9)", fontsize=13, y=1.05, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Visualization] Saved sample grid to {save_path}")

    return fig


def plot_average_digit_intensity(
    X: np.ndarray,
    y: np.ndarray,
    digit: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plots the mean pixel intensity heatmap for a specific digit class.
    Shows where strokes are typically concentrated.
    """
    digit_samples = X[y == digit]
    mean_img = digit_samples.mean(axis=0).reshape(28, 28)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cax = ax.imshow(mean_img, cmap="magma")
    ax.set_title(f"Average Pixel Intensity: Digit '{digit}'", pad=12, fontweight="bold")
    ax.axis("off")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Mean Pixel Intensity [0-1]")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
