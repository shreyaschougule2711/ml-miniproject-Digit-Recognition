"""
Unit tests for multi-digit segmentation and number recognition module.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.multi_digit import (
    preprocess_number_image,
    detect_digit_regions,
    normalize_digit_crop,
    segment_digits,
    predict_multi_digit_number,
)


def _create_synthetic_number_image(text: str, width: int = 400, height: int = 140, bg_white: bool = True) -> np.ndarray:
    """Helper to render synthetic number digits on an image."""
    bg_color = (255, 255, 255) if bg_white else (0, 0, 0)
    fg_color = (0, 0, 0) if bg_white else (255, 255, 255)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw strokes simulating numbers
    # We draw spaced vertical/horizontal lines or digit approximations
    char_spacing = width // (len(text) + 1)
    for idx, char in enumerate(text):
        cx = (idx + 1) * char_spacing
        cy = height // 2
        # Simple simulated digit strokes
        if char == "1":
            draw.line([(cx, cy - 35), (cx, cy + 35)], fill=fg_color, width=12)
        elif char == "0":
            draw.ellipse([cx - 20, cy - 35, cx + 20, cy + 35], outline=fg_color, width=12)
        elif char == "2":
            draw.line([(cx - 20, cy - 35), (cx + 20, cy - 35)], fill=fg_color, width=12)
            draw.line([(cx + 20, cy - 35), (cx - 20, cy + 35)], fill=fg_color, width=12)
            draw.line([(cx - 20, cy + 35), (cx + 20, cy + 35)], fill=fg_color, width=12)
        else:
            draw.rectangle([cx - 15, cy - 35, cx + 15, cy + 35], fill=fg_color)

    return np.array(img)


def test_preprocess_number_image_polarity():
    # White background with black text
    img_white_bg = np.full((100, 200, 3), 255, dtype=np.uint8)
    img_white_bg[40:60, 80:120] = 0

    binary, gray, is_inv = preprocess_number_image(img_white_bg)
    assert is_inv is True
    assert binary.shape == (100, 200)
    assert binary[50, 100] == 255  # Digit stroke is white in binary
    assert binary[10, 10] == 0     # Background is black in binary


def test_detect_digit_regions_ordering():
    # Create binary image with 3 spaced vertical bars (representing digits 1 1 1)
    bin_img = np.zeros((100, 300), dtype=np.uint8)
    bin_img[20:80, 50:60] = 255   # digit at x=50
    bin_img[20:80, 150:160] = 255 # digit at x=150
    bin_img[20:80, 250:260] = 255 # digit at x=250

    boxes, _ = detect_digit_regions(bin_img)
    assert len(boxes) == 3
    # Check left-to-right ordering
    assert boxes[0][0] < boxes[1][0] < boxes[2][0]


def test_normalize_digit_crop():
    # Simple binary digit crop (40x20)
    crop = np.zeros((40, 20), dtype=np.uint8)
    crop[10:30, 8:12] = 255

    crop_28, flat_784 = normalize_digit_crop(crop)
    assert crop_28.shape == (28, 28)
    assert flat_784.shape == (1, 784)
    assert crop_28.min() >= 0.0
    assert crop_28.max() <= 1.0


def test_segment_digits_empty():
    blank = np.full((100, 200, 3), 255, dtype=np.uint8)
    res = segment_digits(blank)
    assert res["success"] is False
    assert "empty" in res["error"] or "blank" in res["error"]


def test_predict_multi_digit_pipeline_with_mock(monkeypatch):
    from sklearn.dummy import DummyClassifier

    # Mock load_model to return a dummy model that predicts 2
    dummy = DummyClassifier(strategy="constant", constant=2)
    dummy.fit(np.zeros((10, 784)), np.full(10, 2))
    monkeypatch.setattr("src.predict.load_model", lambda key: dummy)

    # 3-digit synthetic image
    img = _create_synthetic_number_image("120", width=300, height=100)
    res = predict_multi_digit_number(img, model_key="knn")

    assert res["success"] is True
    assert res["digit_count"] == 3
    assert len(res["predicted_number_str"]) == 3
    assert res["predicted_number_str"] == "222"
    assert res["annotated_image"].shape[:2] == (100, 300)
