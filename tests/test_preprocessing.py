"""
Unit tests for image preprocessing module.
"""

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    normalize_and_flatten,
    is_canvas_blank,
    extract_grayscale_array,
    preprocess_canvas_image,
)


def test_normalize_and_flatten():
    # 2D dummy image batch (10 images of 28x28 with 0-255 uint8)
    raw_images = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
    flattened = normalize_and_flatten(raw_images)

    assert flattened.shape == (10, 784), f"Expected shape (10, 784), got {flattened.shape}"
    assert flattened.dtype == np.float32, f"Expected float32, got {flattened.dtype}"
    assert flattened.min() >= 0.0, "Values below 0.0 detected"
    assert flattened.max() <= 1.0, "Values above 1.0 detected"


def test_is_canvas_blank():
    # Completely black or empty
    blank_canvas = np.zeros((100, 100), dtype=np.float32)
    assert is_canvas_blank(blank_canvas) is True

    # Canvas with a drawn stroke
    drawn_canvas = np.zeros((100, 100), dtype=np.float32)
    drawn_canvas[40:60, 48:52] = 255.0  # stroke representing digit 1
    assert is_canvas_blank(drawn_canvas) is False


def test_extract_grayscale_polarity():
    # Simulate user drawing black stroke (0) on white canvas (255)
    white_bg_canvas = np.full((100, 100, 4), 255, dtype=np.uint8)
    # Draw black stroke in center
    white_bg_canvas[40:60, 48:52, :3] = 0

    gray = extract_grayscale_array(white_bg_canvas)
    # Inverted result should have black background (~0) and white stroke (~255)
    assert gray[0, 0] < 20.0, "Background should be inverted to black"
    assert gray[50, 50] > 200.0, "Drawn stroke should be converted to white"


def test_preprocess_canvas_image_empty():
    # Empty canvas None
    img_28x28, flat_784, err = preprocess_canvas_image(None)
    assert img_28x28 is None
    assert flat_784 is None
    assert err is not None

    # Blank canvas image
    blank = np.zeros((200, 200, 4), dtype=np.uint8)
    img_28x28, flat_784, err = preprocess_canvas_image(blank)
    assert img_28x28 is None
    assert flat_784 is None
    assert "No digit detected" in err or "empty" in err


def test_preprocess_canvas_image_valid_digit():
    # Create white canvas (200x200 RGBA) with a drawn black vertical stroke (digit '1')
    canvas = np.full((200, 200, 4), 255, dtype=np.uint8)
    canvas[50:150, 95:105, :3] = 0  # Stroke

    img_28x28, flat_784, err = preprocess_canvas_image(canvas)
    assert err is None, f"Unexpected error: {err}"
    assert img_28x28 is not None
    assert img_28x28.shape == (28, 28)
    assert flat_784.shape == (1, 784)
    assert flat_784.min() >= 0.0
    assert flat_784.max() <= 1.0
    # Center region should contain positive activation
    assert np.sum(img_28x28[10:18, 10:18]) > 0.0
