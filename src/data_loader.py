"""
MNIST Dataset Loader and Splitter Module.

Handles downloading, caching, inspecting, and splitting the MNIST dataset.
Ensures zero data leakage between training and testing splits.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit

import config


def get_cache_file_path() -> Path:
    """Returns the local npz cache file path for the full MNIST dataset."""
    return config.DATA_DIR / "mnist_784.npz"


def download_or_load_raw_mnist(force_download: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads or loads cached MNIST 784 dataset.
    Returns:
        X (np.ndarray): Shape (70000, 784), uint8 pixel values [0, 255]
        y (np.ndarray): Shape (70000,), int64 digit labels [0, 9]
    """
    cache_path = get_cache_file_path()

    if not force_download and cache_path.exists():
        print(f"[DataLoader] Loading cached MNIST dataset from {cache_path}...")
        try:
            data = np.load(cache_path)
            X = data["X"]
            y = data["y"]
            print(f"[DataLoader] Successfully loaded {X.shape[0]} samples from local cache.")
            return X, y
        except Exception as e:
            print(f"[DataLoader] Failed to load cache ({e}). Re-downloading from OpenML...")

    print("[DataLoader] Downloading MNIST dataset from OpenML (70,000 samples)...")
    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.uint8)
        y = mnist.target.astype(np.int64)

        # Save to local cache for instant future loading
        print(f"[DataLoader] Saving dataset cache to {cache_path}...")
        np.savez_compressed(cache_path, X=X, y=y)
        print("[DataLoader] Cache saved successfully.")
        return X, y
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch MNIST dataset from OpenML. Please check internet connection.\nDetails: {e}"
        )


def get_data_splits(
    train_size: Optional[int] = config.DEFAULT_TRAIN_SAMPLES,
    test_size: Optional[int] = config.DEFAULT_TEST_SAMPLES,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns train and test splits following the official MNIST protocol:
    The official MNIST consists of 60,000 train images and 10,000 test images.
    If train_size < 60,000, a stratified subset is drawn ONLY from the 60,000 train pool.
    If test_size < 10,000, a stratified subset is drawn ONLY from the 10,000 test pool.
    
    This guarantees NO test set leakage into training or hyperparameter selection.

    Returns:
        X_train, y_train, X_test, y_test, metadata_dict
    """
    X_all, y_all = download_or_load_raw_mnist()

    # Official MNIST division: first 60k is train, last 10k is test
    X_train_full = X_all[:60000]
    y_train_full = y_all[:60000]
    X_test_full = X_all[60000:]
    y_test_full = y_all[60000:]

    # Train sampling
    if train_size is not None and train_size < 60000:
        sss_train = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, _ = next(sss_train.split(X_train_full, y_train_full))
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        is_train_subset = True
    else:
        X_train = X_train_full
        y_train = y_train_full
        is_train_subset = False

    # Test sampling
    if test_size is not None and test_size < 10000:
        sss_test = StratifiedShuffleSplit(n_splits=1, train_size=test_size, random_state=random_state)
        test_idx, _ = next(sss_test.split(X_test_full, y_test_full))
        X_test = X_test_full[test_idx]
        y_test = y_test_full[test_idx]
        is_test_subset = True
    else:
        X_test = X_test_full
        y_test = y_test_full
        is_test_subset = False

    metadata = {
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "is_train_subset": is_train_subset,
        "is_test_subset": is_test_subset,
        "features": int(X_train.shape[1]),
        "image_shape": [config.IMAGE_HEIGHT, config.IMAGE_WIDTH],
        "classes": list(range(10)),
        "train_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    }

    print(
        f"[DataLoader] Splits prepared: Train = {X_train.shape[0]} samples (subset={is_train_subset}), "
        f"Test = {X_test.shape[0]} samples (subset={is_test_subset})"
    )
    return X_train, y_train, X_test, y_test, metadata


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, meta = get_data_splits()
    print("DataLoader test run successful:", meta)
