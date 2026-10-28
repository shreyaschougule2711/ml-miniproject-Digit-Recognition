"""
Multi-Digit Number Recognition and Segmentation Pipeline.

Provides end-to-end computer vision and classification for variable-length
handwritten numbers (e.g. 12, 45, 108, 256, 2026, 12345):
1. Input acquisition (canvas drawing or uploaded image).
2. Foreground polarity detection and adaptive/Otsu thresholding.
3. Noise filtering and connected-component / contour analysis.
4. Heuristic merging of broken/multi-stroke digit components.
5. Vertical projection profile fallback for touching digits.
6. Left-to-right spatial sequencing.
7. Aspect-ratio & Center-of-Mass normalization to 28x28 MNIST format.
8. Single & multi-digit classification using trained k-NN, Linear SVM, or RBF SVM models.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift

import config
from src.preprocessing import extract_grayscale_array, center_image_by_mass
from src.predict import predict_digit, load_model


def preprocess_number_image(
    image_input: Union[np.ndarray, Image.Image]
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Converts canvas array or uploaded image into a standardized binary image
    with white foreground digits on a black background.

    Returns:
        binary_img: uint8 array (H, W) where digits are 255 and background is 0.
        grayscale_img: uint8 array (H, W) [0, 255].
        is_inverted: bool indicating if image colors were inverted.
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGBA"))
    else:
        img_np = np.array(image_input)

    # Convert to grayscale
    if img_np.ndim == 3:
        if img_np.shape[2] == 4:
            r, g, b, a = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], img_np[:, :, 3]
            gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
            # If transparent background
            if np.any(a < 250):
                gray = np.where(a > 20, gray, 0.0)
        elif img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = img_np[:, :, 0].astype(np.float32)
    else:
        gray = img_np.astype(np.float32)

    # Polarity detection: Check borders to see if background is white or black
    h, w = gray.shape
    border_pixels = np.concatenate([
        gray[0:max(1, int(h * 0.05)), :].flatten(),
        gray[-max(1, int(h * 0.05)):, :].flatten(),
        gray[:, 0:max(1, int(w * 0.05))].flatten(),
        gray[:, -max(1, int(w * 0.05)):].flatten(),
    ])
    border_mean = np.mean(border_pixels) if len(border_pixels) > 0 else 0

    is_inverted = False
    if border_mean > 127:
        # Dark digits on light background -> Invert to white digits on dark background
        gray = 255.0 - gray
        is_inverted = True

    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)

    # Gaussian blur to reduce high-frequency noise
    blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 0)

    # Otsu thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small morphological cleanup: remove single-pixel specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary, gray_uint8, is_inverted


def _split_touching_digits_if_needed(
    crop_binary: np.ndarray,
    box: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int]]:
    """
    Examines an unusually wide bounding box (aspect ratio width/height > 1.2).
    Uses vertical projection profile (column pixel sums) to find the valley
    and split touching digits.
    """
    x, y, w, h = box
    aspect_ratio = w / float(h)

    # A single digit typically has aspect ratio < 1.1 (most digits are 0.4 to 0.9)
    # If aspect ratio is large, it likely contains 2 or more touching digits
    if aspect_ratio < 1.25 or w < 24:
        return [box]

    # Calculate vertical projection profile (column-wise intensity sum)
    proj = np.sum(crop_binary > 0, axis=0)

    # Search for local minimum in the central region (between 25% and 75% of width)
    search_start = int(w * 0.25)
    search_end = int(w * 0.75)
    if search_end <= search_start:
        return [box]

    central_proj = proj[search_start:search_end]
    min_idx = search_start + int(np.argmin(central_proj))
    min_val = proj[min_idx]

    # Check if this minimum is a true valley (significantly lower than peaks)
    left_peak = np.max(proj[:min_idx]) if min_idx > 0 else 1
    right_peak = np.max(proj[min_idx:]) if min_idx < w else 1
    max_peak = max(left_peak, right_peak)

    if max_peak > 0 and (min_val / max_peak) < 0.65:
        # Valid split point detected!
        split_x = min_idx
        box1 = (x, y, split_x, h)
        box2 = (x + split_x, y, w - split_x, h)
        # Recursively check box2 in case 3 digits are touching
        return [box1] + _split_touching_digits_if_needed(crop_binary[:, split_x:], box2)

    return [box]


def detect_digit_regions(
    binary_img: np.ndarray,
    min_area: int = 40
) -> Tuple[List[Tuple[int, int, int, int]], bool]:
    """
    Detects individual digit bounding boxes using contour analysis and
    connected-component heuristics.
    
    Returns:
        sorted_boxes: List of (x, y, w, h) tuples ordered strictly left-to-right.
        has_touching_split: bool indicating if touching digits were split.
    """
    h_img, w_img = binary_img.shape
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore tiny specks/dots that aren't digit strokes
        if area < min_area and (w < 6 or h < 10):
            continue
        # Ignore full-canvas borders
        if w > 0.98 * w_img and h > 0.98 * h_img:
            continue

        raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return [], False

    # Sort initially by x coordinate
    raw_boxes.sort(key=lambda b: b[0])

    # 1. Merge overlapping or disjoint strokes that belong to the same digit
    # (e.g. detached stroke of 4, 5, 7, dot, or broken loop)
    merged_boxes = []
    skip_indices = set()

    for i in range(len(raw_boxes)):
        if i in skip_indices:
            continue
        x1, y1, w1, h1 = raw_boxes[i]

        for j in range(i + 1, len(raw_boxes)):
            if j in skip_indices:
                continue
            x2, y2, w2, h2 = raw_boxes[j]

            # Horizontal overlap check
            h_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            min_w = min(w1, w2)

            # Horizontal gap
            h_gap = x2 - (x1 + w1) if x2 > (x1 + w1) else 0

            # Vertical proximity
            v_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

            # Merge condition: substantial horizontal overlap or tiny horizontal gap with vertical overlap
            should_merge = False
            if min_w > 0 and (h_overlap / min_w) > 0.5:
                should_merge = True
            elif h_gap < 6 and v_overlap > 0:
                # Disjoint components very close together
                should_merge = True

            if should_merge:
                # Merge into enclosing box
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                x1, y1, w1, h1 = new_x, new_y, new_w, new_h
                skip_indices.add(j)

        merged_boxes.append((x1, y1, w1, h1))

    # 2. Split touching digits using vertical projection profile
    final_boxes = []
    has_touching_split = False

    for box in merged_boxes:
        x, y, w, h = box
        crop_bin = binary_img[y : y + h, x : x + w]
        split_result = _split_touching_digits_if_needed(crop_bin, box)
        if len(split_result) > 1:
            has_touching_split = True
        final_boxes.extend(split_result)

    # 3. Sort final boxes strictly from left to right
    final_boxes.sort(key=lambda b: b[0])

    return final_boxes, has_touching_split


def normalize_digit_crop(crop_gray_or_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a single segmented digit crop into the standard 28x28 MNIST format:
    - Preserves aspect ratio inside a 20x20 bounding box.
    - Centers the digit inside 28x28 using Center of Mass (LeCun standard).
    - Scales pixel values to [0.0, 1.0] float32.

    Returns:
        crop_28x28: shape (28, 28) float32 values [0.0, 1.0]
        flattened_784: shape (1, 784) float32 values [0.0, 1.0]
    """
    crop = crop_gray_or_bin.astype(np.float32)
    crop_h, crop_w = crop.shape

    # Protect against tiny single dots
    if crop_h < 2 or crop_w < 2:
        empty_28 = np.zeros((28, 28), dtype=np.float32)
        return empty_28, empty_28.reshape(1, 784)

    # Scale to 20x20 preserving aspect ratio
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20)))

    pil_cropped = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    pil_resized = pil_cropped.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_np = np.array(pil_resized, dtype=np.float32)

    # Embed in 28x28 canvas
    canvas_28x28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28x28[start_y : start_y + new_h, start_x : start_x + new_w] = resized_np

    # Center by mass
    centered_28x28 = center_image_by_mass(canvas_28x28)

    # Normalize to [0.0, 1.0]
    normalized_28x28 = np.clip(centered_28x28 / 255.0, 0.0, 1.0).astype(np.float32)
    flattened_784 = normalized_28x28.reshape(1, 784)

    return normalized_28x28, flattened_784


def segment_digits(
    image_input: Union[np.ndarray, Image.Image]
) -> Dict[str, Any]:
    """
    Performs full segmentation on an image containing one or more handwritten digits.

    Returns:
        result dictionary with:
        - success (bool)
        - digit_count (int)
        - boxes (list of (x, y, w, h))
        - digit_crops (list of (28, 28) arrays)
        - flattened_digits (list of (1, 784) arrays)
        - annotated_image (np.ndarray RGB with drawn bounding boxes)
        - binary_image (np.ndarray)
        - has_touching_split (bool)
        - error (str or None)
    """
    if image_input is None:
        return {
            "success": False,
            "error": "No input image provided.",
            "digit_count": 0,
            "boxes": [],
            "digit_crops": [],
            "flattened_digits": [],
        }

    binary_img, gray_img, is_inverted = preprocess_number_image(image_input)

    # Check if empty canvas / image
    if np.sum(binary_img > 0) < 15:
        return {
            "success": False,
            "error": "Image is empty or blank. Please draw or upload handwritten numbers.",
            "digit_count": 0,
            "boxes": [],
            "digit_crops": [],
            "flattened_digits": [],
        }

    boxes, has_touching_split = detect_digit_regions(binary_img)

    if not boxes:
        return {
            "success": False,
            "error": "No distinct digit regions detected. Please write clearly with visible strokes.",
            "digit_count": 0,
            "boxes": [],
            "digit_crops": [],
            "flattened_digits": [],
        }

    # Generate annotated image
    annotated = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    digit_crops = []
    flattened_digits = []

    for idx, (x, y, w, h) in enumerate(boxes, 1):
        # Crop from the GRAYSCALE image (not binary) to preserve antialiased
        # gradients that match the smooth pixel values in MNIST training data.
        # Binary thresholding destroys edge gradients, causing misclassifications
        # (e.g. 5→6) across all classifiers.
        pad = 2
        y1 = max(0, y - pad)
        y2 = min(gray_img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray_img.shape[1], x + w + pad)

        crop_gray = gray_img[y1:y2, x1:x2]
        crop_28x28, flat_784 = normalize_digit_crop(crop_gray)

        digit_crops.append(crop_28x28)
        flattened_digits.append(flat_784)

        # Draw green bounding box on annotated image
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (37, 99, 235), 2)
        # Draw badge with digit index
        badge_text = f"#{idx}"
        cv2.putText(
            annotated,
            badge_text,
            (x, max(14, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (16, 185, 129),
            2,
            cv2.LINE_AA
        )

    return {
        "success": True,
        "error": None,
        "digit_count": len(boxes),
        "boxes": boxes,
        "digit_crops": digit_crops,
        "flattened_digits": flattened_digits,
        "annotated_image": annotated,
        "binary_image": binary_img,
        "grayscale_image": gray_img,
        "has_touching_split": has_touching_split,
    }


def predict_multi_digit_number(
    image_input: Union[np.ndarray, Image.Image],
    model_key: str = "rbf_svm"
) -> Dict[str, Any]:
    """
    Complete end-to-end multi-digit recognition pipeline:
    1. Segments input into ordered digit regions.
    2. Runs each digit through the selected classifier.
    3. Combines predictions left-to-right into full number string (preserving leading zeros).
    4. Computes individual and average confidence metrics.

    Returns:
        result dictionary containing:
        - success (bool)
        - predicted_number_str (str, e.g. "2026")
        - digit_count (int)
        - digits (list of individual prediction dicts)
        - average_confidence (float)
        - model_key (str)
        - annotated_image (np.ndarray RGB)
        - error (str or None)
        - warning (str or None)
    """
    seg_res = segment_digits(image_input)

    if not seg_res["success"]:
        return {
            "success": False,
            "error": seg_res["error"],
            "predicted_number_str": "",
            "digit_count": 0,
            "digits": [],
            "average_confidence": 0.0,
        }

    digit_results = []
    predicted_chars = []
    confidences = []

    for idx, flat_784 in enumerate(seg_res["flattened_digits"], 1):
        pred_single = predict_digit(model_key, flat_784)
        if not pred_single.get("success"):
            return {
                "success": False,
                "error": f"Digit #{idx} prediction failed: {pred_single.get('error')}",
                "predicted_number_str": "",
                "digit_count": 0,
                "digits": [],
                "average_confidence": 0.0,
            }

        pred_digit = pred_single["predicted_digit"]
        conf = pred_single["confidence"]
        predicted_chars.append(str(pred_digit))
        confidences.append(conf)

        digit_results.append({
            "index": idx,
            "box": seg_res["boxes"][idx - 1],
            "crop_28x28": seg_res["digit_crops"][idx - 1],
            "predicted_digit": pred_digit,
            "confidence": conf,
            "confidence_percentage": pred_single["confidence_percentage"],
            "score_type": pred_single["score_type"],
            "top_3": pred_single["top_3"],
            "all_scores": pred_single["all_scores"],
            "explanation": pred_single["explanation"],
        })

    # Preserve leading zeros by joining as string
    predicted_number_str = "".join(predicted_chars)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    warning_msg = None
    if seg_res.get("has_touching_split"):
        warning_msg = (
            "Adjacent digits appeared to be touching. Vertical projection profiling was used to "
            "separate them. For maximum accuracy, maintain natural spacing between digits."
        )

    return {
        "success": True,
        "error": None,
        "warning": warning_msg,
        "model_key": model_key,
        "predicted_number_str": predicted_number_str,
        "digit_count": len(digit_results),
        "digits": digit_results,
        "average_confidence": round(avg_confidence, 4),
        "average_confidence_percentage": round(avg_confidence * 100, 2),
        "annotated_image": seg_res["annotated_image"],
        "binary_image": seg_res["binary_image"],
        "grayscale_image": seg_res.get("grayscale_image"),
    }
