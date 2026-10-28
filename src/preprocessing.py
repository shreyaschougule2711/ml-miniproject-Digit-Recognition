"""
Preprocessing Pipeline Module for MNIST & User-Drawn Canvas Input.

Provides:
1. Normalization and flattening functions for dataset arrays.
2. Centering, aspect-ratio preserving resizing, and polarity correction for drawn canvas.
3. Center-of-mass shift adhering to the official Yann LeCun MNIST creation protocol.
"""

from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift


def normalize_and_flatten(X: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel intensities from [0, 255] to [0.0, 1.0] and ensures
    shape is (N, 784) with float32 precision.
    """
    X_float = X.astype(np.float32)
    if X_float.max() > 1.0:
        X_float = X_float / 255.0
    if X_float.ndim > 2:
        X_float = X_float.reshape(X_float.shape[0], -1)
    return X_float


def is_canvas_blank(img_array: np.ndarray, threshold: float = 5.0) -> bool:
    """
    Checks if the canvas image is empty / blank.
    Expects grayscale array where foreground (drawing) has intensity > threshold.
    """
    return bool(np.sum(img_array > threshold) < 10)


def extract_grayscale_array(canvas_input: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Converts canvas input (which could be RGBA numpy array from streamlit-drawable-canvas
    or a PIL Image) into a 2D float32 grayscale array [0, 255], ensuring the digit
    stroke is bright (white) on a dark (black) background, just like MNIST.
    """
    if isinstance(canvas_input, Image.Image):
        # Convert PIL Image to RGBA or L
        if canvas_input.mode != "RGBA":
            canvas_input = canvas_input.convert("RGBA")
        img_np = np.array(canvas_input)
    else:
        img_np = np.array(canvas_input)

    # If 3D array (Height, Width, Channels)
    if img_np.ndim == 3:
        if img_np.shape[2] == 4:
            # RGBA: If user drew with white pen on black background or black pen on white
            r, g, b, a = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2], img_np[:, :, 3]
            # Standard luminance
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            
            # If alpha channel was used to draw (transparent background)
            if np.any(a > 0) and np.any(a < 255):
                # Mask out transparent background
                gray = np.where(a > 10, gray, 0.0)
        elif img_np.shape[2] == 3:
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = img_np[:, :, 0]
    else:
        gray = img_np.astype(np.float32)

    # Check background polarity:
    # In MNIST: background is black (~0), digit is white (~255).
    # If canvas background is white (borders/corners mean > 128), invert it!
    corner_pixels = np.concatenate([
        gray[:10, :10].flatten(),
        gray[:10, -10:].flatten(),
        gray[-10:, :10].flatten(),
        gray[-10:, -10:].flatten()
    ])
    if np.mean(corner_pixels) > 128:
        # Invert black-on-white to white-on-black
        gray = 255.0 - gray

    # Clip and convert to float32
    gray = np.clip(gray, 0.0, 255.0).astype(np.float32)
    return gray


def center_image_by_mass(img_28x28: np.ndarray) -> np.ndarray:
    """
    Centers the 28x28 image using its Center of Mass, matching the official
    MNIST preprocessing methodology created by Yann LeCun and colleagues.
    """
    cy, cx = center_of_mass(img_28x28)
    if np.isnan(cx) or np.isnan(cy):
        return img_28x28

    # Target center in 28x28 is (13.5, 13.5) or approx (14, 14)
    shift_x = np.round(14.0 - cx).astype(int)
    shift_y = np.round(14.0 - cy).astype(int)

    # Limit shift to avoid shifting digit off canvas
    shift_x = np.clip(shift_x, -5, 5)
    shift_y = np.clip(shift_y, -5, 5)

    centered = shift(img_28x28, shift=(shift_y, shift_x), mode="constant", cval=0.0)
    return centered


def preprocess_canvas_image(canvas_input: Union[np.ndarray, Image.Image]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Processes user-drawn canvas image into MNIST standard representation:
    1. Grayscale conversion with polarity check (white digit on black background).
    2. Check if empty/blank canvas.
    3. Crop tight bounding box around the digit.
    4. Maintain aspect ratio and resize to fit within a 20x20 box.
    5. Place inside a 28x28 canvas with 4-pixel padding.
    6. Center using Center of Mass (standard MNIST protocol).
    7. Normalize to [0.0, 1.0] float32.

    Returns:
        processed_28x28 (np.ndarray of shape (28, 28) with values in [0, 1] or None if blank)
        flattened_784 (np.ndarray of shape (1, 784) with values in [0, 1] or None if blank)
        error_msg (str or None)
    """
    if canvas_input is None:
        return None, None, "Canvas is empty. Please draw a digit."

    gray = extract_grayscale_array(canvas_input)

    # Check if empty canvas
    if is_canvas_blank(gray, threshold=15.0):
        return None, None, "No digit detected. Please draw a digit (0-9) on the canvas."

    # Detect bounding box of the digit (pixels with intensity > threshold)
    threshold = 20.0
    y_indices, x_indices = np.where(gray > threshold)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None, None, "No digit detected. Please draw more distinctly."

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Crop digit
    cropped = gray[y_min : y_max + 1, x_min : x_max + 1]
    crop_h, crop_w = cropped.shape

    # Protect against tiny single dots
    if crop_h < 4 and crop_w < 4:
        return None, None, "Drawing is too small. Please draw a clearer digit."

    # Fit into 20x20 box preserving aspect ratio
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20)))

    pil_cropped = Image.fromarray(cropped.astype(np.uint8))
    pil_resized = pil_cropped.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_np = np.array(pil_resized, dtype=np.float32)

    # Pad into 28x28 canvas
    canvas_28x28 = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas_28x28[start_y : start_y + new_h, start_x : start_x + new_w] = resized_np

    # Center by mass
    centered_28x28 = center_image_by_mass(canvas_28x28)

    # Normalize to [0.0, 1.0]
    normalized_28x28 = centered_28x28 / 255.0
    normalized_28x28 = np.clip(normalized_28x28, 0.0, 1.0).astype(np.float32)

    flattened_784 = normalized_28x28.reshape(1, 784)

    return normalized_28x28, flattened_784, None
